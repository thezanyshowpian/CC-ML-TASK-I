import os
import joblib
import numpy as np
import pandas as pd
import warnings

class LiveAuctionAgent:
    """
    Pawn Stars Autonomous Bidding Agent
    =====================================
    Strategy Summary:
    -----------------
    Cars arrive one at a time sequentially. There is no future car to
    save budget for — every pass is a permanently missed win. Strategy
    is tuned to maximize both wins and profit simultaneously, with
    wins weighted slightly higher (20% vs 15% in grading).

    CORE PHILOSOPHY:
        High confidence prediction → maximize PROFIT per win
            Prediction is reliable, be selective and patient.
            Only enter if lower bound guarantees profit.
            Demand 2% margin. even with this margin the profit per win will still be high
            as our model predicts them better. we added the margin 2% so that entry condition dont 
            stop our profit.

        Low confidence prediction → maximize WIN COUNT
            Prediction is uncertain, prioritize getting on the board while avoiding losses.
            Enter if mean shows any positive profit.
            Accept 10% margin — a thin win still counts.

    1. ENTRY DECISION  — Confidence-split entry gates.
                         High confidence (width < 0.139):
                             enter only if lower bound > current bid (strict)
                         Low confidence (width 0.139-0.414):
                             enter if mean shows any positive profit (lenient)
                         Pass: mean shows no profit, model failure,
                               extreme uncertainty, or already overpriced.

    2. BID CEILING     — Confidence-tiered anchor + margin.
                         High confidence → anchor to mean,   margin 10%
                         Low confidence  → anchor to lower,  margin 5%
                         Thresholds from EDA CV structural breaks at condition 3.0

    3. POSITION SIZING — Fractional Kelly Criterion.
                         kelly_f = edge / variance
                         safe_f  = kelly_f x (0.5 x bankroll_ratio)
                         High confidence: hard cap 10% of bankroll
                         Low confidence:  40% of Kelly, hard cap 4%
                         As bankroll depletes, Kelly fraction shrinks automatically.

    4. INCREMENT SIZING — Hyperbolic decay sequence.
                         decay          = 1 / (1 + round x 0.5)
                         conf_penalty   = relative_width x 0.30
                         adjusted_decay = decay x (1 - conf_penalty)
                         increment      = headroom x safe_f x adjusted_decay
                         Uncertain cars decay faster, exit sooner if margin gone.
                         Maintains credible bids throughout — rivals cannot read
                         budget exhaustion from collapsing increment sizes.

    5. WINNER'S CURSE  — Bid shading via lower bound anchoring.
                         Winning reveals you were the most optimistic bidder.
                         Using 25th percentile as ceiling naturally shades bids
                         below the mean — implementing auction theory without
                         explicit opponent modeling.

    6. NEGATIVE PREDICTIONS — Principled abstention, not a bug.
                         38 validation cars had negative mean predictions.
                         Actual median price was $375 — negligible margin.
                         Passing costs nothing; bidding blindly risks real loss.
    """

    # -------------------------------------------------------------------------
    # Thresholds — all derived from EDA, not arbitrary
    # -------------------------------------------------------------------------
    MIN_PRICE            = 100
    HIGH_CONF_MARGIN     = 0.05
    LOW_CONF_MARGIN      = 0.1
    WIDTH_HIGH_CONF      = 0.139
    WIDTH_PASS           = 0.414
    KELLY_BASE_FRACTION  = 0.50
    MAX_BANKROLL_RISK    = 0.10
    LOW_CONF_BANKROLL    = 0.04
    DECAY_RATE           = 0.50
    CONFIDENCE_DECAY     = 0.30
    MIN_INCREMENT        = 50.0

    def __init__(self):

        self.bankroll       = 500000.0
        self.starting_bank  = 500000.0

        self.predicted_value = 0.0
        self.lower_bound     = 0.0
        self.upper_bound     = 0.0
        self.relative_width  = 1.0
        self.ceiling         = 0.0
        self.current_round   = 0
        self.high_confidence = False

        self.wins            = 0
        self.total_profit    = 0.0

        base_path = os.path.dirname(os.path.abspath(__file__))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model    = joblib.load(os.path.join(base_path, "model_MdAdnanKhalid.pkl"))
            self.encoders = joblib.load(os.path.join(base_path, "encoders_MdAdnanKhalid.pkl"))

        self.lower_model      = self.encoders['lower_quantile_model']
        self.upper_model      = self.encoders['upper_quantile_model']
        self.target_encoder   = self.encoders['target_encoder']
        self.tx_rules         = self.encoders['transmission']
        self.odo_rules        = self.encoders['odometer']
        self.cond_rules       = self.encoders['condition']
        self.color_rules      = self.encoders['color']
        self.interior_rules   = self.encoders['interior']
        self.body_rules       = self.encoders['body']
        self.body_mapping     = self.encoders['body_mapping']
        self.reference_year   = self.encoders['reference_year']
        self.FEATURE_COLUMNS  = self.encoders['feature_cols']
        self.ohe_columns      = self.encoders['ohe_columns']

        self.LUXURY_MAKES      = self.encoders['luxury_makes']
        self.EXOTIC_COLORS     = self.encoders['exotic_colors']
        self.PREMIUM_INTERIORS = self.encoders['premium_interiors']

        self.BODY_CATEGORIES = [
            col.replace('body_clean_', '')
            for col in self.ohe_columns
            if col.startswith('body_clean_')
        ]

    # =========================================================================
    # IMPUTATION HELPERS
    # =========================================================================

    def _impute_transmission(self, car: dict) -> str:
        val = car.get('transmission', '')
        if not val or str(val).strip() in ('', 'nan', 'none'):
            make  = car.get('make', '')
            model = car.get('model', '')
            rules = self.tx_rules
            if (make, model) in rules['make_model']:
                return rules['make_model'][(make, model)]
            elif make in rules['make']:
                return rules['make'][make]
            return rules['global']
        return str(val).strip()

    def _impute_numeric(self, car: dict, col: str) -> float:
        val = car.get(col)
        try:
            if val is not None and not pd.isna(float(val)):
                return float(val)
        except (TypeError, ValueError):
            pass
        rules = self.odo_rules if col == 'odometer' else self.cond_rules

        # FIXED — cast year to int to match integer keys in rules dict
        try:
            year = int(float(car.get('year'))) if car.get('year') is not None else None
        except (TypeError, ValueError):
            year = None

        model = car.get('model', '')
        if year is not None and (year, model) in rules['year_model'] and pd.notna(rules['year_model'][(year, model)]):
            return float(rules['year_model'][(year, model)])
        elif year is not None and year in rules['year'] and pd.notna(rules['year'][year]):
            return float(rules['year'][year])
        return float(rules['global'])

    def _impute_categorical(self, car: dict, col: str) -> str:
        val = car.get(col)
        if not val or str(val).strip() in ('', 'nan', 'none'):
            rules = (self.color_rules    if col == 'color'    else
                     self.interior_rules if col == 'interior' else
                     self.body_rules)
            make  = car.get('make', '')
            model = car.get('model', '')
            trim  = car.get('trim', '')
            if (make, model, trim) in rules['make_model_trim']:
                return rules['make_model_trim'][(make, model, trim)]
            elif (make, model) in rules['make_model']:
                return rules['make_model'][(make, model)]
            return rules['global']
        return str(val).strip()

    # =========================================================================
    # PREPROCESSING PIPELINE
    # =========================================================================

    def _preprocess(self, raw: dict) -> np.ndarray:

        # Step 1 — Lowercase all strings
        car = {
            k: (v.lower().strip() if isinstance(v, str) else v)
            for k, v in raw.items()
        }

        # Step 2 — Fix impossible odometer before imputation
        try:
            odo = float(car.get('odometer', 0))
            if odo < 10 or odo > 900000:
                car['odometer'] = None
        except (TypeError, ValueError):
            car['odometer'] = None

        # Step 3 — Impute all missing values
        car['transmission'] = self._impute_transmission(car)
        car['odometer']     = self._impute_numeric(car, 'odometer')
        car['condition']    = self._impute_numeric(car, 'condition')
        color               = self._impute_categorical(car, 'color')
        interior            = self._impute_categorical(car, 'interior')
        body_raw            = self._impute_categorical(car, 'body')

        # Step 4 — Binary feature flags
        make        = car.get('make', '')
        is_luxury   = int(make in self.LUXURY_MAKES)
        has_exotic  = int(color in self.EXOTIC_COLORS)
        has_premium = int(interior in self.PREMIUM_INTERIORS)
        lux_exotic  = int(bool(is_luxury) and bool(has_exotic))
        lux_premium = int(bool(is_luxury) and bool(has_premium))

        # Step 5 — Target encoding via fitted TargetEncoder
        encode_df = pd.DataFrame([{
            'make' : make,
            'model': car.get('model', ''),
            'trim' : car.get('trim', ''),
            'state': car.get('state', '')
        }])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoded = self.target_encoder.transform(encode_df)

        make_enc  = float(encoded['make'].iloc[0])
        model_enc = float(encoded['model'].iloc[0])
        trim_enc  = float(encoded['trim'].iloc[0])
        state_enc = float(encoded['state'].iloc[0])

        # Step 6 — OHE: transmission
        tx_manual = int(car.get('transmission', 'automatic') == 'manual')

        # Step 7 — OHE: body_clean
        body_clean = self.body_mapping.get(body_raw, 'other')
        body_ohe   = {
            f'body_clean_{b}': int(body_clean == b)
            for b in self.BODY_CATEGORIES
        }

        # Step 8 — Assemble feature dict
        # FIXED — use imputed car values, not raw car.get() with 0 fallback
        feature_dict = {
            'year'                   : float(car.get('year', self.reference_year)),
            'make'                   : make_enc,
            'model'                  : model_enc,
            'trim'                   : trim_enc,
            'state'                  : state_enc,
            'condition'              : float(car['condition']),
            'odometer'               : float(car['odometer']),
            'is_luxury'              : is_luxury,
            'has_exotic_color'       : has_exotic,
            'has_premium_interior'   : has_premium,
            'luxury_exotic_color'    : lux_exotic,
            'luxury_premium_interior': lux_premium,
            'transmission_manual'    : tx_manual,
            **body_ohe
        }

        return np.array([[feature_dict[col] for col in self.FEATURE_COLUMNS]])

    # =========================================================================
    # ANALYZE ITEM
    # =========================================================================

    def analyze_item(self, item_features: dict):

        self.current_round   = 0
        self.high_confidence = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X = self._preprocess(item_features)

            mean_pred  = float(self.model.predict(X)[0])
            lower_pred = float(self.lower_model.predict(X)[0])
            upper_pred = float(self.upper_model.predict(X)[0])

        if mean_pred > 0:
            mean_pred = max(self.MIN_PRICE, mean_pred)
        if lower_pred > 0:
            lower_pred = max(self.MIN_PRICE, lower_pred)
        if upper_pred > 0:
            upper_pred = max(self.MIN_PRICE, upper_pred)

        if lower_pred > upper_pred or lower_pred <= 0 or upper_pred <= 0:
            lower_pred = mean_pred * 0.85
            upper_pred = mean_pred * 1.15

        relative_width = (upper_pred - lower_pred) / max(mean_pred, 1)
        relative_width = max(0.0, min(relative_width, self.WIDTH_PASS))

        self.high_confidence = relative_width < self.WIDTH_HIGH_CONF

        # Determine the Anchor
        if self.high_confidence:
            anchor = mean_pred
        else:
            anchor = lower_pred
            
        # Dynamic Margin Percentage based on asset value and confidence
        # ---------------------------------------------------------
        # ADVERSARIAL MARGIN STRATEGY (For Smart Competitors)
        # ---------------------------------------------------------
        if mean_pred < 10000:
            # Drop from 30% to 15%. 
            # Smart agents know these are cheap, so bids will be tighter.
            # We rely on our lower_bound (25th percentile) model to protect us here.
            margin_pct = 0.15 
            
        elif mean_pred > 30000:
            # Top-tier assets. Smart agents will fight to the death for these.
            # We compress to 1.5% to guarantee we edge them out, relying on 
            # the natural dollar-value buffer of a $40k car.
            margin_pct = 0.015 
            
        else:
            # Mid-tier combat zone. 
            if self.high_confidence:
                margin_pct = self.HIGH_CONF_MARGIN  # 5% (Down from 10%)
            else:
                margin_pct = self.LOW_CONF_MARGIN  # 10% (Demand more safety if our model is unsure)   # 10%

        # Apply the margin to set the bidding ceiling
        self.predicted_value = mean_pred
        self.lower_bound     = lower_pred
        self.upper_bound     = upper_pred
        self.relative_width  = relative_width
        self.ceiling         = anchor * (1.0 - margin_pct)

    # =========================================================================
    # PLACE BID
    # =========================================================================

    def place_bid(self, current_highest_bid: float) -> float:

        if self.predicted_value < self.MIN_PRICE:
            return 0.0

        if self.relative_width >= self.WIDTH_PASS:
            return 0.0

        if self.ceiling <= current_highest_bid:
            return 0.0

        if current_highest_bid >= self.bankroll:
            return 0.0

        edge      = (self.lower_bound     - current_highest_bid) / max(current_highest_bid, 1.0)
        mean_edge = (self.predicted_value - current_highest_bid) / max(current_highest_bid, 1.0)

        if self.high_confidence:
            if edge <= 0:
                return 0.0
        else:
            if mean_edge <= 0:
                return 0.0

        kelly_edge     = edge if self.high_confidence else mean_edge
        variance       = max(self.relative_width ** 2, 1e-6)
        kelly_f        = kelly_edge / variance
        bankroll_ratio = self.bankroll / self.starting_bank
        safe_f         = kelly_f * (self.KELLY_BASE_FRACTION * bankroll_ratio)

        if self.high_confidence:
            safe_f = min(safe_f, self.MAX_BANKROLL_RISK)
        else:
            safe_f = min(safe_f * 0.40, self.LOW_CONF_BANKROLL)

        safe_f = max(safe_f, 0.001)

        max_exposure   = self.bankroll * safe_f
        effective_ceil = min(self.ceiling, current_highest_bid + max_exposure)

        if effective_ceil <= current_highest_bid:
            return 0.0

        headroom       = effective_ceil - current_highest_bid
        decay          = 1.0 / (1.0 + self.current_round * self.DECAY_RATE)
        conf_penalty   = self.relative_width * self.CONFIDENCE_DECAY
        adjusted_decay = decay * (1.0 - conf_penalty)
        adjusted_decay = max(adjusted_decay, 0.05)

        increment      = headroom * safe_f * adjusted_decay
        increment      = max(increment, self.MIN_INCREMENT)

        next_bid       = current_highest_bid + increment
        self.current_round += 1

        if next_bid > effective_ceil:
            return 0.0

        if next_bid > self.bankroll:
            return 0.0

        return round(next_bid, 2)

    # =========================================================================
    # AUCTION RESULT
    # =========================================================================

    def auction_result(self, won: bool, winning_bid: float,
                       actual_price: float, current_bankroll: float):

        self.bankroll = current_bankroll

        if won:
            self.wins        += 1
            expected_profit   = self.predicted_value - winning_bid
            self.total_profit += expected_profit