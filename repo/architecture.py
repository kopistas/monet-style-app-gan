#!/usr/bin/env python3
"""
Generate an architecture diagram for the Monet Style Transfer infrastructure.
This script uses the diagrams library to create a simple architecture diagram.

Install requirements: pip install diagrams
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Kubernetes
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.mlops import Mlflow
from diagrams.onprem.network import Nginx
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack
from diagrams.generic.device import Mobile

# Create the diagram
with Diagram("Monet Style Transfer Architecture", show=False, filename="architecture", outformat="png"):
    
    # Users
    users = Mobile("Users")
    
    # GitHub
    with Cluster("GitHub"):
        github_repo = GithubActions("CI/CD Pipeline")
    
    # DigitalOcean
    with Cluster("DigitalOcean"):
        
        # S3 Storage
        spaces = Storage("DO Spaces\n(S3 Storage)")
        
        # Kubernetes Cluster
        with Cluster("Kubernetes Cluster (DOKS)"):
            k8s = Kubernetes("Kubernetes\nControl Plane")
            
            with Cluster("Networking"):
                ingress = Nginx("Ingress\nController")
            
            with Cluster("MLflow Namespace"):
                mlflow = Mlflow("MLflow Server")
            
            with Cluster("Web App Namespace"):
                web_app = Flask("Monet Style\nTransfer App")
                
    # Google Colab
    with Cluster("Google Colab"):
        colab = Python("Model Training")
    
    # Connect everything
    users >> ingress
    ingress >> web_app
    ingress >> mlflow
    
    web_app >> spaces
    mlflow >> spaces
    
    colab >> mlflow
    colab >> spaces
    
    github_repo >> k8s
    
    # Bidirectional connections
    mlflow - Edge(style="dotted") - web_app
    
if __name__ == "__main__":
    print("Architecture diagram generated: architecture.png") 