import config

class RiskEngine:
    """
    Rule-Based Risk Score Engine (0-100).
    NOTE: This is a monitoring risk indicator, NOT a probability of cheating.
    
    Rule-Based Scoring Breakdown:
    - Prohibited object              : +25
    - Persistent prohibited object   : +15
    - Object handling                : +20
    - Repeated head movement         : +10
    - Possible object passing        : +25
    - Maximum                        : 100
    
    Risk Levels:
    - 0–29   : NORMAL
    - 30–59  : ATTENTION
    - 60–100 : HIGH RISK
    """

    def __init__(self):
        self.score_prohibited_object = config.RISK_PROHIBITED_OBJECT
        self.score_persistent_prohibited_object = config.RISK_PERSISTENT_PROHIBITED_OBJECT
        self.score_object_handling = config.RISK_OBJECT_HANDLING
        self.score_head_movement = config.RISK_REPEATED_HEAD_MOVEMENT
        self.score_object_passing = config.RISK_POSSIBLE_OBJECT_PASSING

    def calculate_risk(
        self,
        has_prohibited_object: bool = False,
        is_persistent: bool = False,
        is_object_handling: bool = False,
        has_head_movement: bool = False,
        has_object_passing: bool = False
    ) -> dict:
        """
        Calculates rule-based monitoring risk score (0-100), risk level indicator, and breakdown.
        
        Returns:
            dict containing:
                - risk_score (int): 0-100
                - risk_level (str): 'NORMAL' | 'ATTENTION' | 'HIGH RISK'
                - breakdown (list): List of triggered rule point contributions
        """
        total_score = 0
        breakdown = []

        # Rule 1: Prohibited object (+25)
        if has_prohibited_object:
            total_score += self.score_prohibited_object
            breakdown.append({"rule": "Prohibited object", "points": self.score_prohibited_object})

        # Rule 2: Persistent prohibited object (+15)
        if is_persistent:
            total_score += self.score_persistent_prohibited_object
            breakdown.append({"rule": "Persistent prohibited object", "points": self.score_persistent_prohibited_object})

        # Rule 3: Object handling (+20)
        if is_object_handling:
            total_score += self.score_object_handling
            breakdown.append({"rule": "Object handling", "points": self.score_object_handling})

        # Rule 4: Repeated head movement (+10)
        if has_head_movement:
            total_score += self.score_head_movement
            breakdown.append({"rule": "Repeated head movement", "points": self.score_head_movement})

        # Rule 5: Possible object passing (+25)
        if has_object_passing:
            total_score += self.score_object_passing
            breakdown.append({"rule": "Possible object passing", "points": self.score_object_passing})

        # Maximum = 100
        final_score = min(100, max(0, total_score))

        # Risk Indicator Thresholds:
        # 0–29    NORMAL
        # 30–59   ATTENTION
        # 60–100  HIGH RISK
        if final_score <= 29:
            risk_level = "NORMAL"
        elif final_score <= 59:
            risk_level = "ATTENTION"
        else:
            risk_level = "HIGH RISK"

        return {
            "risk_score": final_score,
            "risk_level": risk_level,
            "is_cheating_probability": False,
            "breakdown": breakdown
        }

    def calculate_incident_risk(self, incident_data: dict) -> dict:
        """
        Backward compatibility wrapper for incident payload dictionaries.
        """
        cat = incident_data.get("category", "")
        event_type = incident_data.get("event_type", "").lower()
        obj_name = str(incident_data.get("object", incident_data.get("object_name", ""))).lower()
        duration = incident_data.get("duration", incident_data.get("duration_seconds", 0.0))

        has_prohibited = (cat == "RESTRICTED" or cat == "PROHIBITED" or obj_name in ["phone", "earbuds", "smartwatch"] or "prohibited" in event_type)
        is_persistent = has_prohibited and duration >= 3.0
        is_handling = "handling" in event_type
        has_head = "head movement" in event_type
        has_passing = "passing" in event_type

        res = self.calculate_risk(
            has_prohibited_object=has_prohibited,
            is_persistent=is_persistent,
            is_object_handling=is_handling,
            has_head_movement=has_head,
            has_object_passing=has_passing
        )
        res["contributing_factors"] = res["breakdown"]
        return res

if __name__ == "__main__":
    re = RiskEngine()
    print("Testing RiskEngine Step 11...")
    
    # Test case 1: Clean exam frame
    clean_res = re.calculate_risk()
    print("Clean Frame -> Score:", clean_res["risk_score"], "| Level:", clean_res["risk_level"])
    
    # Test case 2: Phone detected (+25)
    phone_res = re.calculate_risk(has_prohibited_object=True)
    print("Prohibited Object -> Score:", phone_res["risk_score"], "| Level:", phone_res["risk_level"])
    
    # Test case 3: Phone + Persistent + Handling + Head Movement (+25 +15 +20 +10 = 70)
    high_res = re.calculate_risk(
        has_prohibited_object=True,
        is_persistent=True,
        is_object_handling=True,
        has_head_movement=True
    )
    print("High Risk Session -> Score:", high_res["risk_score"], "| Level:", high_res["risk_level"])
