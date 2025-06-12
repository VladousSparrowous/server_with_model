# server_with_model
model selection and server

The project consisted of two consecutive phases:

Data Science & Machine Learning – Preprocessing, training and evaluating models(Ridge, Lasso, Random Forest, lightGBM) on a provided dataset to select the best-performing one based on specified metrics.

DevOps/Backend – Implementing a lightweight service to expose the selected model via an HTTP API, allowing users to send feature inputs and receive predictions(using FastAPI).

start
```
pip install -r requirements.txt
```

Requests to the server:

GET

/ping

```
curl -X 'GET' \
  'http://localhost/ping' \
  -H 'accept: application/json' 
```

response

```
{
  "status": "ok",
  "total_queries": 1,
  "successful_queries": 1
}
```

POST

/inference

```
curl -X 'POST' \
  'http://localhost/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "Promo_start": "2024-07-24",
    "Promo_end": "2024-07-30",
    "Shipping_start": "2024-06-25",
    "Shipping_end": "2024-07-30",
    "Promo_type": "J",
    "Feat_2": 10328.21,
    "Feat_3": 58.22,
    "Agent": "C",
    "Promo_id": "Promo №5483.0",
    "Item_id": "Item ID: 125.0",
    "Feat_7": 32270.51,
    "Promo_class": "D",
    "Feat_9": 84812245.76,
    "Feat_10": 5718813.46,
    "Feat_11": 25.96,
    "Feat_12": 84212
}'
```

response

```
{
  "prediction": 1591.773194901201,
  "model": "random_forest"
}
```
