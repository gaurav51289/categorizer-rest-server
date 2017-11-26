## REST API server for categorizer

### Start

1. `python run.py`
------------------

### Endpoint

| Endpoints       | Body                                     |
|-----------------|------------------------------------------|
| `POST /categorize` | **{question : 'Whatever is your question?'}**|

### cURL

`curl -d '{"question":"How to install python on Ubuntu 16.04?"}' -H "Content-Type: application/json" -X POST http://localhost:5001/categorize/`