"""Extract explicit recommendation scores; no paper-quality correctness gold."""
import json
import re

ALLOWED_ANCHORS={1,3,5,6,8,10}


def score(text,item):
    text=text if isinstance(text,str) else ''
    cleaned=text.replace('**','').replace('`','')
    values=[];evidence=[]
    # Same-line explicitly named field only; never consume a rubric list or
    # a correctness/confidence rating, and never infer a score from accept prose.
    pattern=r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?Recommendation(?:[ \t]+(?:Score|Rating))?[ \t]*:[ \t]*(\d+(?:\.\d+)?)(?:[ \t]*/[ \t]*10)?(?=\s|:|[.,;]|$)[^\n]*$'
    for m in re.finditer(pattern,cleaned):values.append(float(m[1]));evidence.append(m[0].strip())
    # A separate header may be followed by a single standalone numeric line.
    pattern=r'(?im)^\s*(?:#{1,6}\s*)?Recommendation(?:\s+(?:Score|Rating))?\s*:?\s*\n\s*(\d+(?:\.\d+)?)(?:\s*/\s*10)?\s*$'
    for m in re.finditer(pattern,cleaned):values.append(float(m[1]));evidence.append(m[0].strip())
    try:
        obj=json.loads(text)
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key.lower() in ('recommendation','recommendation_score') and isinstance(value,(int,float,str)) and not isinstance(value,bool):
                    if re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',str(value)):values.append(float(value));evidence.append({key:value})
    except (ValueError,TypeError):pass
    valid=bool(values) and len(set(values))==1 and values[0].is_integer() and 1<=values[0]<=10
    parsed=int(values[0]) if valid else None
    status=('explicit_recommendation' if valid else ('conflicting_recommendations' if len(set(values))>1 else ('out_of_range_or_noninteger' if values else ('empty' if not text.strip() else 'missing_explicit_recommendation'))))
    return {'correct':None,'parsed':parsed,'recommendation':parsed,'parse_status':status,
            'score_evidence':evidence,'matches_reconstructed_allowed_anchor':parsed in ALLOWED_ANCHORS if valid else None,
            'recommends_accept_at_threshold6':parsed>=6 if valid else None,
            'quality_gold_available':False,'new_review_expert_deficiency_gold_available':False}
