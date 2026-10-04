import yaml
import csv
import requests
import argparse
import statistics
import logging
import sys
import datetime
import os
import json

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger("edgeclaw")

# Creating a benchmark standalone function which takes yaml file, endpoints, 
# csv file and write in csv file
def run_benchmark(queries_path : str, server_url: str, out_dir: str) -> dict :

    #Load the yaml file data
    with open(queries_path, "r", encoding="utf-8") as f :
        data = yaml.safe_load(f)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"results_{timestamp}.csv")
    json_path = os.path.join(out_dir, f"results_{timestamp}.json")

    url = server_url + "/chat"
    health = server_url + "/health"
    rows = []

    #Check for health of LLM , getting response from /health endpoint
    try : 
        resp = requests.get(health)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Connection health not good, error : {e}")
        print("Error message", file=sys.stderr)
        return {}

    # Getting resp from /chat endpoint and adding details to csv_dict
    for entry in data :
        payload = {"query" : entry["query"]}

        try:
            response = requests.post(url, json=payload)  # /chat endpoint expects json body
            response.raise_for_status()
            body = response.json()
        except Exception as e:
            logger.warning(f"Couldnt send post request, exception : {e}")
            csv_dict = {
                        "id" : entry["id"],
                        "query" : entry["query"],
                        "latency_ms": None,
                        "num_context_chunks" : None,
                        "backend" : None,
                        "model" : None,
                        "total_keywords" : len(entry["keywords"]),
                        "keywords_matched" : 0,
                        "answer_preview" :  f"<ERROR: {str(e)[:120]}>",
                        "route" : None,
                        "route_reason" : None,
                        "fallback_used" : None,
                        "fallback_reason" : None
                    }
            rows.append(csv_dict)
            continue

        keywords_matched = 0

        #Appending info to rows of csv file via csv_dict
        ans_lower = body["answer"].lower()
        for keyword in entry["keywords"] : 
            if keyword.lower() in ans_lower:
                keywords_matched += 1
        csv_dict = {
            "id" : entry["id"],
            "query" : entry["query"],
            "latency_ms": body["latency_ms"],
            "num_context_chunks" : body["num_context_chunks"],
            "backend" : body["backend"],
            "model" : body["model"],
            "total_keywords" : len(entry["keywords"]),
            "keywords_matched" : keywords_matched,
            "answer_preview" : body["answer"][:120],
            "route" : body["route"],
            "route_reason" : body["route_reason"],
            "fallback_used" : body.get("fallback_used"),
            "fallback_reason" : body.get("fallback_reason")
        }
        rows.append(csv_dict)
    
    # Writing the values in csv file
    with open(csv_path, 'w' , newline = '',  encoding="utf-8") as csvfile :
        fieldnames = ["id", "query", "latency_ms", "num_context_chunks", "backend", "model", 
                      "total_keywords", "keywords_matched", "answer_preview", "route", "route_reason",
                      "fallback_used", "fallback_reason"]
        writer = csv.DictWriter(csvfile, fieldnames= fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_payload = {
        "timestamp" : timestamp,
        "server_url" : server_url,
        "num_rows" : len(rows),
        "rows" : rows
    }

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_payload, json_file, indent=2, ensure_ascii=False)

    latency = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    if len(latency) == 0 :
        avg_latency_ms = None
    else :
        avg_latency_ms = statistics.mean(latency)
    fallback_count = 0
    for row in rows :
        if row.get("fallback_used") is True :
            fallback_count += 1
    return {"rows" : len(rows), "avg_latency_ms" : avg_latency_ms , "fallback_count": fallback_count, "csv": csv_path, "json": json_path }
    
# Logic for calling run_benchmark standalone
if __name__ == "__main__":
    parser= argparse.ArgumentParser(description="Run Edgeclaw benchmark against a running server")
    parser.add_argument("--queries" , default="benchmarks/queries.yaml", 
                        help="Path to YAML file with test queries")
    parser.add_argument("--server", default = "http://127.0.0.1:8000", 
                        help="Base URL of the running EdgeClaw server")
    parser.add_argument("--out", default = "benchmarks/results/", 
                        help="Path where the results CSV and JSON will be written")
    args = parser.parse_args()
    rows_dict = run_benchmark(args.queries, args.server, args.out)

    if not rows_dict:
        print("Benchmark Aborted, check the error above")
    else : 
        print(f"rows written: {rows_dict['rows']} with avg latency: {rows_dict['avg_latency_ms']}, "
              f" fallbacks : {rows_dict['fallback_count']} to csv file : {rows_dict['csv']}")
