#!/usr/bin/env python3
"""
Verify the deployment status of Kubernetes resources for the Monet application.

This script checks:
1. Pod status in both namespaces
2. Service availability
3. Ingress configuration
4. Endpoint connectivity

It requires the kubectl to be configured with the proper context.
"""

import os
import subprocess
import json
import time
import sys
import requests
from urllib.parse import urlparse

def run_command(command):
    """Run a shell command and return the output"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def check_pods(namespace):
    """Check if pods in the namespace are running"""
    print(f"\n=== Checking pods in {namespace} namespace ===")
    
    # Get all pods in the namespace
    pods_json = run_command(f"kubectl get pods -n {namespace} -o json")
    if not pods_json:
        print(f"  ERROR: Failed to get pods in {namespace} namespace")
        return False
    
    pods = json.loads(pods_json)
    if not pods.get('items'):
        print(f"  WARNING: No pods found in {namespace} namespace")
        return False
    
    all_running = True
    for pod in pods['items']:
        name = pod['metadata']['name']
        status = pod['status']['phase']
        ready_containers = 0
        total_containers = 0
        
        if 'containerStatuses' in pod['status']:
            for container in pod['status']['containerStatuses']:
                total_containers += 1
                if container.get('ready'):
                    ready_containers += 1
        
        ready_status = f"{ready_containers}/{total_containers}" if total_containers > 0 else "0/0"
        
        print(f"  Pod: {name} - Status: {status} - Ready: {ready_status}")
        
        if status != "Running" or ready_containers != total_containers:
            all_running = False
            
            # Check for container errors
            if 'containerStatuses' in pod['status']:
                for container in pod['status']['containerStatuses']:
                    if 'waiting' in container['state'] and 'reason' in container['state']['waiting']:
                        reason = container['state']['waiting']['reason']
                        message = container['state']['waiting'].get('message', 'No message')
                        print(f"    Container {container['name']} is waiting: {reason} - {message}")
    
    if all_running:
        print(f"  SUCCESS: All pods in {namespace} namespace are running")
    else:
        print(f"  WARNING: Some pods in {namespace} namespace are not ready")
        # Get pod logs
        for pod in pods['items']:
            name = pod['metadata']['name']
            print(f"\n  === Logs for {name} ===")
            logs = run_command(f"kubectl logs -n {namespace} {name} --tail=50")
            print(f"  {logs}")
    
    return all_running

def check_services(namespace):
    """Check if services in the namespace are available"""
    print(f"\n=== Checking services in {namespace} namespace ===")
    
    # Get all services in the namespace
    services_json = run_command(f"kubectl get services -n {namespace} -o json")
    if not services_json:
        print(f"  ERROR: Failed to get services in {namespace} namespace")
        return False
    
    services = json.loads(services_json)
    if not services.get('items'):
        print(f"  WARNING: No services found in {namespace} namespace")
        return False
    
    for service in services['items']:
        name = service['metadata']['name']
        service_type = service['spec']['type']
        ports = service['spec']['ports']
        port_info = ", ".join([f"{p.get('name', 'unnamed')}:{p.get('port')}->{p.get('targetPort')}" for p in ports])
        
        print(f"  Service: {name} - Type: {service_type} - Ports: {port_info}")
        
        # Check if service has endpoints
        endpoints_json = run_command(f"kubectl get endpoints -n {namespace} {name} -o json")
        if not endpoints_json:
            print(f"    WARNING: Failed to get endpoints for service {name}")
            continue
        
        endpoints = json.loads(endpoints_json)
        if not endpoints.get('subsets'):
            print(f"    WARNING: No endpoints found for service {name}")
            continue
        
        for subset in endpoints['subsets']:
            addresses = subset.get('addresses', [])
            if not addresses:
                print(f"    WARNING: No addresses found for service {name}")
                continue
            
            print(f"    Endpoints: {len(addresses)} addresses")
    
    print(f"  SUCCESS: Services in {namespace} namespace checked")
    return True

def check_ingress(namespace):
    """Check ingress configuration"""
    print(f"\n=== Checking ingress in {namespace} namespace ===")
    
    # Get all ingresses in the namespace
    ingress_json = run_command(f"kubectl get ingress -n {namespace} -o json")
    if not ingress_json:
        print(f"  ERROR: Failed to get ingress in {namespace} namespace")
        return None
    
    ingresses = json.loads(ingress_json)
    if not ingresses.get('items'):
        print(f"  WARNING: No ingress found in {namespace} namespace")
        return None
    
    hosts = []
    for ingress in ingresses['items']:
        name = ingress['metadata']['name']
        rules = ingress['spec'].get('rules', [])
        
        print(f"  Ingress: {name}")
        
        if not rules:
            print(f"    WARNING: No rules found for ingress {name}")
            continue
        
        for rule in rules:
            host = rule.get('host')
            if not host:
                print(f"    WARNING: No host found in rule")
                continue
            
            hosts.append(host)
            print(f"    Host: {host}")
            
            paths = rule.get('http', {}).get('paths', [])
            if not paths:
                print(f"    WARNING: No paths found for host {host}")
                continue
            
            for path in paths:
                path_type = path.get('pathType')
                path_value = path.get('path')
                backend = path.get('backend', {})
                service_name = backend.get('service', {}).get('name')
                service_port = backend.get('service', {}).get('port', {}).get('number')
                
                print(f"    Path: {path_value} ({path_type}) -> {service_name}:{service_port}")
    
    return hosts

def check_endpoint_connectivity(hosts):
    """Check if the endpoints are accessible"""
    print("\n=== Checking endpoint connectivity ===")
    
    if not hosts:
        print("  No hosts to check")
        return False
    
    all_ok = True
    for host in hosts:
        url = f"https://{host}"
        print(f"  Checking {url}...")
        
        try:
            response = requests.get(url, timeout=10, verify=False)
            status = response.status_code
            print(f"  Response status: {status}")
            
            if status >= 200 and status < 400:
                print(f"  SUCCESS: Endpoint {url} is accessible")
            elif status == 503:
                print(f"  ERROR: Endpoint {url} returns 503 Service Unavailable")
                print("  This usually means the backend service is not ready or misconfigured")
                all_ok = False
            else:
                print(f"  WARNING: Endpoint {url} returns status {status}")
                all_ok = False
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Failed to connect to {url}: {str(e)}")
            all_ok = False
    
    return all_ok

def check_nginx_ingress_logs():
    """Check logs from the NGINX ingress controller"""
    print("\n=== Checking NGINX ingress controller logs ===")
    
    # Find the ingress controller pod
    pods_json = run_command("kubectl get pods -n ingress-nginx -l app.kubernetes.io/component=controller -o json")
    if not pods_json:
        print("  ERROR: Failed to find NGINX ingress controller pod")
        return False
    
    pods = json.loads(pods_json)
    if not pods.get('items'):
        print("  WARNING: No NGINX ingress controller pods found")
        return False
    
    controller_pod = pods['items'][0]['metadata']['name']
    print(f"  Found ingress controller pod: {controller_pod}")
    
    # Get logs with errors
    print("  Recent error logs:")
    logs = run_command(f"kubectl logs -n ingress-nginx {controller_pod} | grep -i error | tail -20")
    if logs:
        print(f"  {logs}")
    else:
        print("  No error logs found")
    
    return True

def check_load_balancer():
    """Check the load balancer IP"""
    print("\n=== Checking Load Balancer ===")
    
    # Get the load balancer service
    service_json = run_command("kubectl get service -n ingress-nginx nginx-ingress-ingress-nginx-controller -o json")
    if not service_json:
        print("  ERROR: Failed to get NGINX ingress controller service")
        return None
    
    service = json.loads(service_json)
    lb_status = service.get('status', {}).get('loadBalancer', {}).get('ingress', [])
    
    if not lb_status:
        print("  WARNING: Load balancer does not have an IP assigned yet")
        return None
    
    lb_ip = lb_status[0].get('ip')
    if lb_ip:
        print(f"  Load balancer IP: {lb_ip}")
        return lb_ip
    else:
        print("  WARNING: Load balancer does not have an IP")
        return None

def wait_for_deployments(namespaces, timeout=300):
    """Wait for deployments to be available"""
    print("\n=== Waiting for deployments to be available ===")
    
    start_time = time.time()
    all_available = False
    
    while time.time() - start_time < timeout:
        all_available = True
        
        for namespace in namespaces:
            print(f"  Checking deployments in {namespace} namespace...")
            
            # Get all deployments in the namespace
            deployments_json = run_command(f"kubectl get deployments -n {namespace} -o json")
            if not deployments_json:
                print(f"  ERROR: Failed to get deployments in {namespace} namespace")
                all_available = False
                continue
            
            deployments = json.loads(deployments_json)
            if not deployments.get('items'):
                print(f"  WARNING: No deployments found in {namespace} namespace")
                continue
            
            for deployment in deployments['items']:
                name = deployment['metadata']['name']
                replicas = deployment['spec'].get('replicas', 0)
                available = deployment['status'].get('availableReplicas', 0)
                
                print(f"  Deployment: {name} - Replicas: {available}/{replicas}")
                
                if available < replicas:
                    all_available = False
        
        if all_available:
            print("  SUCCESS: All deployments are available")
            return True
        
        print(f"  Waiting for deployments to be available... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(10)
    
    print(f"  ERROR: Timeout waiting for deployments to be available ({timeout}s)")
    return False

def main():
    """Main entry point"""
    print("Starting deployment verification...")
    
    # Check kubectl configuration
    kubeconfig = os.environ.get('KUBECONFIG')
    print(f"Using KUBECONFIG: {kubeconfig}")
    
    # Check if kubectl is available
    version = run_command("kubectl version --client")
    if not version:
        print("ERROR: kubectl not found or not configured correctly")
        sys.exit(1)
    
    # Wait for deployments to be available
    namespaces = ['monet-app', 'mlflow']
    if not wait_for_deployments(namespaces):
        sys.exit(1)
    
    # Check load balancer
    lb_ip = check_load_balancer()
    
    # Check each namespace
    all_pods_running = True
    hosts = []
    
    for namespace in namespaces:
        # Check pods
        if not check_pods(namespace):
            all_pods_running = False
        
        # Check services
        check_services(namespace)
        
        # Check ingress
        namespace_hosts = check_ingress(namespace)
        if namespace_hosts:
            hosts.extend(namespace_hosts)
    
    # Check NGINX ingress controller logs if there are issues
    if not all_pods_running:
        check_nginx_ingress_logs()
    
    # Check endpoint connectivity
    if hosts:
        check_endpoint_connectivity(hosts)
    
    print("\n=== Summary ===")
    if all_pods_running:
        print("✅ All pods are running")
    else:
        print("❌ Some pods are not running")
    
    if lb_ip:
        print(f"✅ Load balancer IP: {lb_ip}")
    else:
        print("❌ Load balancer IP not assigned")
    
    if hosts:
        print(f"✅ Configured hosts: {', '.join(hosts)}")
    else:
        print("❌ No hosts configured")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main() 