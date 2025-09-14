# Explanation of BRK:: Fields in Mentor-Mentee Matching

Below is a breakdown of each `BRK::` field, what it means, and how to interpret the values in your analysis tables:

---

## 1. `BRK::pref_gender`
- **Meaning:**  
  Points awarded if the mentor matches the mentee’s **preferred gender**.
- **Interpretation:**  
  - If the value is equal to the configured weight (e.g., 30), the preference is satisfied.
  - If 0, the mentor does **not** match the mentee’s preferred gender.

---

## 2. `BRK::pref_eth`
- **Meaning:**  
  Points awarded if the mentor matches the mentee’s **preferred ethnicity**.
- **Interpretation:**  
  - If the value is the full weight (e.g., 30), the preference is satisfied.
  - If 0, the mentor does **not** match the mentee’s preferred ethnicity.

---

## 3. `BRK::mentor_pref_gender`
- **Meaning:**  
  Points (usually half weight) if the mentee matches the **mentor’s preferred gender**.
- **Interpretation:**  
  - If nonzero (e.g., 15), the mentee matches the mentor’s preference.
  - If 0, the mentee does **not** match the mentor’s preferred gender.

---

## 4. `BRK::mentor_pref_eth`
- **Meaning:**  
  Points (usually half weight) if the mentee matches the **mentor’s preferred ethnicity**.
- **Interpretation:**  
  - If nonzero (e.g., 15), the mentee matches the mentor’s preference.
  - If 0, the mentee does **not** match the mentor’s preferred ethnicity.

---

## 5. `BRK::location_state`
- **Meaning:**  
  Points awarded if mentor and mentee are in the **same state**.
- **Interpretation:**  
  - If the value is the configured weight (e.g., 15), they are in the same state.
  - If 0, they are in different states.

---

## 6. `BRK::same_school`
- **Meaning:**  
  Points awarded if mentor and mentee are from the **same school**.
- **Interpretation:**  
  - If the value is the configured weight (e.g., 20), they are from the same school.
  - If 0, they are from different schools.

---

## 7. `BRK::year_proximity`
- **Meaning:**  
  Points awarded if mentor and mentee’s **class years are within 2 years** of each other.
- **Interpretation:**  
  - If the value is the configured weight (e.g., 20), their class years are close.
  - If 0, their class years are not close.

---

## 8. `BRK::hobby_overlap_exact`
- **Meaning:**  
  Points for **each exact hobby** that both mentor and mentee share.
- **Interpretation:**  
  - Value = (number of shared hobbies) × (weight per hobby, e.g., 10).
  - If 0, no exact hobby overlap.

---

## 9. `BRK::mbti`
- **Meaning:**  
  Points based on **MBTI compatibility** using a predefined matrix.
- **Interpretation:**  
  - Higher values mean more compatible MBTI types.
  - Range depends on the MBTI matrix (e.g., 0–15).

---

## 10. `BRK::goals_rule`
- **Meaning:**  
  Points for **rule-based similarity** between mentor and mentee goals (e.g., keyword overlap or related sets).
- **Interpretation:**  
  - 8 = exact match, 4 = related, 0 = not related.

---

## 11. `BRK::goals_semantic`
- **Meaning:**  
  Points for **semantic similarity** between mentor and mentee goals, using sentence embeddings.
- **Interpretation:**  
  - Higher values mean more semantically similar goals.
  - Range: 0 (no similarity) to the configured max (e.g., 20).

---

## 12. `BRK::interests_semantic`
- **Meaning:**  
  Points for **semantic similarity** between mentor and mentee interests/hobbies, using sentence embeddings.
- **Interpretation:**  
  - Higher values mean more semantically similar interests.
  - Range: 0 (no similarity) to the configured max (e.g., 10).

---

## How to Use These Fields

- **Higher values** in each field mean a stronger match on that specific criterion.
- **0 means no match** for that criterion.
- The **total score** is the sum of all these breakdowns, showing overall compatibility.
- You can use these fields to understand **why** a particular mentor/mentee pair scored high or low, and to audit or explain the matching decisions.