import logging
from app import backend_registry

logger = logging.getLogger("edgeclaw")

# Fallback dispatching logic if any route fails
def dispatch(route_label: str , prompt : str) -> dict :
    chain = backend_registry.get_fallback_chain(route_label)    # List of fallback routes
    attempted = []
    failed = []

    for i, label in enumerate(chain) :
        attempted.append(label)
        try : 
            backend = backend_registry.get_backend(label)   # backend is not available
            if not backend.is_available() :
                fail_msg = f"Backend {label} is not available"
                logger.warning(fail_msg)
                failed.append(fail_msg)
                continue
            result = backend.generate(prompt)   # -> result is BackendResponse object
            if result.error :       # result generated some error
                fail_msg = f"Backend {label} Failed with error : {result.error}"
                logger.warning(fail_msg)
                failed.append(fail_msg)
                continue 
            # All is good, we take this label and return
            final_route = label         
            fallback_used = i > 0
            chain_used = list(attempted)    
            fallback_reason = "; ".join(failed) if failed else None
            return {"result": result, "final_route": final_route, "chain_used": chain_used, 
                    "fallback_used": fallback_used, "fallback_reason": fallback_reason}    

        except Exception as e :
            fail_msg = f"Backend {label} exception : {e}"
            logger.exception(fail_msg)
            failed.append(fail_msg) 
            continue

    # All the labels failed 
    msg = f"All the labels failed, attempted : {attempted} , failed : {failed}"
    logger.error(msg)
    raise RuntimeError(msg)



    
