import yaml
import json
import os
import logging
import argparse
import sys
import glob

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger("edgeclaw")

# Evaluating the answer given by agent
def evaluate(eval_path : str, results_json_path: str) -> dict :

    eval_by_id = {}
    scored, passed, failed, skipped = 0,0,0,0 
    per_row = []
    by_category = {}

    # load yaml file data
    with open(eval_path, "r", encoding="utf-8") as f :
        data_lst = yaml.load(f, Loader=yaml.SafeLoader)

    with open(results_json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    rows_lst = data_json["rows"]

    for vals in data_lst :
        eval_by_id[vals["id"]] = vals

    for row in rows_lst :
        answer = row.get("answer_full") or row.get("answer_preview")
        row_id = row["id"]
        
        if row["id"] not in eval_by_id :    # Skipped row as eval doesnt exist
            skipped += 1
            logger.warning(f"Row id {row['id']} doesn't exist in eval path")
            per_row.append({"id": row_id, "category": None, "verdict": "skipped",
                            "must_contain_misses": [], "must_not_hits": [],
                            "answer": "" if answer is None else answer[:200]})
            continue
        
        if answer is None or answer.startswith("<ERROR"): # Failed as no ans to benchmark
            verdict = "fail"
            eval_entry = eval_by_id[row_id]
            category = eval_entry["category"]
            failed += 1
            scored += 1
            _tally(by_category, category, verdict)   
            per_row.append({"id": row_id, "category": eval_entry["category"], "verdict": verdict,
                            "must_contain_misses": [], "must_not_hits": [],
                            "answer": "" if answer is None else answer[:200]})

        else :  # All good 
            eval_entry = eval_by_id[row_id]
            category = eval_entry["category"]
            answer_lower = answer.lower()
            must_contain_misses = []
            must_not_hits = []

            for word in eval_entry["must_contain"]:   # Words which should be present but are not in ans
                if word.lower() not in answer_lower :
                    must_contain_misses.append(word)
            for word in eval_entry["must_not_contain"] :  # Words which should not be present but are in ans
                if word.lower() in answer_lower :
                    must_not_hits.append(word)
            if len(must_contain_misses) == 0 and len(must_not_hits) == 0 :
                verdict = "pass"
                passed +=1
                scored += 1
                _tally(by_category, category, verdict)                                               
            else :
                verdict = "fail"
                failed += 1
                scored += 1
                _tally(by_category, category, verdict)                    

            per_row.append({"id": row_id, "category": eval_entry["category"], "verdict": verdict,
                            "must_contain_misses": must_contain_misses, "must_not_hits": must_not_hits,
                            "answer": answer[:200]}) 

    if scored > 0 :
        pass_rate = passed/scored
        pass_rate = round(pass_rate*100, 1)
    else : 
        pass_rate = 0.0

    # ---- Pretty-print report ----
    print("\nEdgeClaw Lite — eval report")
    print(f"  eval file   : {eval_path}")
    print(f"  results file: {results_json_path}")
    print()
    print(f"  scored  : {scored}")
    print(f"  passed  : {passed}")
    print(f"  failed  : {failed}")
    print(f"  skipped : {skipped}")
    print(f"  pass rate: {pass_rate}%")
    print()
    if by_category:
        print("  by category:")
        for cat in sorted(by_category):
            stats = by_category[cat]
            s_, p_ = stats["scored"], stats["passed"]
            pct = round(100 * p_ / s_, 1) if s_ > 0 else 0.0
            print(f"    {cat:12} : {p_}/{s_} ({pct}%)")
        print()
    print("  details:")
    for r in per_row:
        parts = [f"    [{r['verdict']:7}]  {r['id']:16}  ({r['category']})"]
        if r["must_contain_misses"]:
            parts.append(f"missing: {r['must_contain_misses']}")
        if r["must_not_hits"]:
            parts.append(f"hallucinated: {r['must_not_hits']}")
        print("  ".join(parts))
    print()
    
    return {"scored": scored, "passed": passed, "failed": failed, "skipped": skipped,
            "pass_rate": pass_rate, "by_category" : by_category, "per_row": per_row}

# Func to get dict with category-wise verdict(pass, fail and scored)
def _tally(by_category, category, verdict):
    if category not in by_category:
        by_category[category] = {"scored" : 0, "passed" : 0}
    by_category[category]["scored"] += 1
    if verdict == "pass" :
        by_category[category]["passed"] += 1

if __name__ == "__main__" :
    parser= argparse.ArgumentParser(description="Evaluation on benchmark")
    parser.add_argument("--eval" , default="benchmarks/eval.yaml", 
                        help="Path to YAML file with ")
    parser.add_argument("--results", default = None, 
                        help="Give the results of evaluation")
    parser.add_argument("--out", default = None, 
                        help="Path where the results CSV and JSON will be written")
    args = parser.parse_args()

    if args.results is None:
        candidate = glob.glob("benchmarks/results/*.json")
        if not candidate :
            print("No benchmark results found. Run `python -m benchmarks.run_benchmark` first.", file=sys.stderr)
            sys.exit(1)

        args.results = max(candidate, key=os.path.getmtime)
    summary = evaluate(args.eval , args.results)

    if args.out is not None :
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent= 2, ensure_ascii= False)