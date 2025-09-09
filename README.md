# Databricks e2e llmops project
## Instructions
- Clone the repo to your folder (locally or in a Databricks git folder)
- Update in databricks.yml
  - Target `host` for each environment
  - `catalog_name`
  - `schema_name`
  - `model_name`
- Deploy
- Run Jobs
  - model_preprocessing 
  - model_build_evaluation
  - model_endpoint_deploy
  - model_inference