import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

def main(args):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    if os.getenv("MLFLOW_USERNAME") and os.getenv("MLFLOW_PASSWORD"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_PASSWORD")

    model_file = os.path.abspath(args.model_file)
    model_name = args.model_name
    aliases = args.aliases or []
    artifact_subpath = args.artifact_path or "models"

    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    client = MlflowClient()
    mlflow.set_experiment("manual_uploads")

    # === Check if alias already exists and has the same file ===
    for alias in aliases:
        try:
            existing_version = client.get_model_version_by_alias(model_name, alias)
            run_id = existing_version.run_id
            files = client.list_artifacts(run_id, path=artifact_subpath)
            filenames = [f.path.split("/")[-1] for f in files]
            expected_name = os.path.basename(model_file)

            if expected_name in filenames:
                print(f"Alias '{alias}' already points to a version with '{expected_name}'. Skipping upload.")
                return
            else:
                print(f"Alias '{alias}' exists but file differs. Proceeding with upload.")
        except RestException:
            print(f"Alias '{alias}' not found. Will create new one.")

    # === Upload and register ===
    with mlflow.start_run(run_name="register_base_model") as run:
        mlflow.log_artifact(model_file, artifact_path=artifact_subpath)
        artifact_uri = f"runs:/{run.info.run_id}/{artifact_subpath}"
        print(f"Registering model from: {artifact_uri}")

        registered = mlflow.register_model(model_uri=artifact_uri, name=model_name)
        print(f"Registered as '{model_name}', version {registered.version}")

        client.update_registered_model(
            name=model_name,
            description="Base pretrained CycleGAN model for Monet style transfer"
        )

        for alias in aliases:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=registered.version
            )
            print(f"Alias '{alias}' set on version {registered.version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path to .pt file")
    parser.add_argument("--model-name", required=True, help="Registered model name")
    parser.add_argument("--aliases", nargs="+", help="List of aliases to assign (e.g. base prod)")
    parser.add_argument("--artifact-path", default="models", help="Artifact folder path in MLflow")
    args = parser.parse_args()
    main(args)
