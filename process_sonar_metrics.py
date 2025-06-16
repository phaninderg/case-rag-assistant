import sys
import re
import requests

# Static mapping of repository names to owners
REPO_OWNERS = {
    'app-collab-chat-core': 'Kireeti Vallabhu',
    'app-dlir': 'Kireeti Vallabhu',
    'app-escm-shift-handover': 'Phaninder Gattu',
    'app-escm-workspace-commons': 'Phaninder Gattu',
    'app-fe-sharepoint-connector': 'Kireeti Vallabhu',
    'app-file-explorer-core': 'Kireeti Vallabhu',
    'app-secops-arcsight-esm': 'Kireeti Vallabhu',
    'app-secops-aws-securityhub': 'Kireeti Vallabhu',
    'app-secops-azure-sentinel': 'Kireeti Vallabhu',
    'app-secops-common': 'Phaninder Gattu',
    'app-secops-crowdstrike-falcon-insight': 'Kireeti Vallabhu',
    'app-secops-crowdstrike-falcon-sandbox': 'Kireeti Vallabhu',
    'app-secops-defender-endpoint': 'Kireeti Vallabhu',
    'app-secops-event-ingestion-common': 'Kireeti Vallabhu',
    'app-secops-fireeyehx': 'Kireeti Vallabhu',
    'app-secops-gen-ai': 'Uday Kumar Bijanepalle',
    'app-secops-graphsecurityapi': 'Kireeti Vallabhu',
    'app-secops-icap-dlp': 'Kireeti Vallabhu',
    'app-secops-microsoft-dlp': 'Kireeti Vallabhu',
    'app-secops-misp': 'Kireeti Vallabhu',
    'app-secops-netskope-dlp': 'Kireeti Vallabhu',
    'app-secops-proofpoint': 'Kireeti Vallabhu',
    'app-secops-proofpoint-dlp': 'Kireeti Vallabhu',
    'app-secops-qradar': 'Kireeti Vallabhu',
    'app-secops-recommended-actions': 'Uday Kumar Bijanepalle',
    'app-secops-report-common': 'Phaninder Gattu',
    'app-secops-secureworks': 'Kireeti Vallabhu',
    'app-secops-symantec-dlp': 'Kireeti Vallabhu',
    'app-secops-threat-intel-security-center': 'Uday Kumar Bijanepalle',
    'app-secops-zscaler': 'Kireeti Vallabhu',
    'app-sir-analyst-workspace': 'Phaninder Gattu',
    'app-sir-urp-ml': 'Phaninder Gattu',
    'app-tisc-int-crowdstrike-host': 'Uday Kumar Bijanepalle',
    'app-tisc-int-defender-edr': 'Uday Kumar Bijanepalle',
    'sn-slack-connector': 'Kireeti Vallabhu'
}

def extract_repo_name(path):
    # Extract repo name from the path field.
    # This pattern ensures the path is exactly 'source/repoName' with nothing after
    match = re.fullmatch(r'source/([^/]+)/?$', path)
    return match.group(1) if match else None

def fetch_sonar_metrics(base_url, component_key, branch, metrics, username, password):
    """Fetch metrics from SonarQube API"""
    metrics_str = ','.join(metrics)
    url = f"{base_url}/api/measures/component_tree"
    
    params = {
        'component': component_key,
        'branch': branch,
        'metricKeys': metrics_str,
        'qualifiers': 'DIR',
        'ps': 500
    }
    
    try:
        response = requests.get(
            url,
            params=params,
            auth=(username, password)
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from SonarQube API: {e}")
        sys.exit(1)

def process_sonar_metrics(data, output_file):
    # List to store metrics for each repo
    repo_metrics = []
    
    # Process each component
    for component in data.get('components', []):
        path = component.get('path', '')
        repo_name = extract_repo_name(path)
        
        if not repo_name or repo_name not in REPO_OWNERS:
            continue
            
        # Initialize metrics for this repo
        metrics = {
            'repoName': repo_name,
            'owner': REPO_OWNERS[repo_name],
            'lines_to_cover': 0,
            'uncovered_lines': 0,
            'covered_lines': 0,
            'coverage': 0.0
        }
        
        # Extract metrics
        for measure in component.get('measures', []):
            metric = measure.get('metric')
            value = measure.get('value')
            
            if not value:
                continue
                
            if metric == 'lines_to_cover':
                metrics['lines_to_cover'] = int(float(value))
            elif metric == 'uncovered_lines':
                metrics['uncovered_lines'] = int(float(value))
            elif metric == 'coverage':
                metrics['coverage'] = float(value)
        
        # Calculate covered lines
        metrics['covered_lines'] = metrics['lines_to_cover'] - metrics['uncovered_lines']
        
        # Add to our list
        repo_metrics.append(metrics)
    
    # Write to CSV
    with open(output_file, 'w') as f:
        # Write header
        f.write('repoName,owner,lines_to_cover,uncovered_lines,covered_lines,coverage\n')
        
        # Write data
        for metrics in sorted(repo_metrics, key=lambda x: x['repoName']):
            f.write(f"{metrics['repoName']},\"{metrics['owner']}\",{metrics['lines_to_cover']},{metrics['uncovered_lines']},{metrics['covered_lines']},{metrics['coverage']:.2f}\n")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_sonar_metrics.py <username> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    
    base_url = 'https://metrics.devsnc.com'
    component_key = 'com.snc:snc-sonar-app-sir'
    branch = 'track/simdev'
    metrics = ['coverage', 'uncovered_lines', 'lines_to_cover']
    output_file = 'sonar_metrics_summary.csv'
    
    print(f"Fetching metrics from {base_url}...")
    data = fetch_sonar_metrics(base_url, component_key, branch, metrics, username, password)
    print("Processing metrics...")
    process_sonar_metrics(data, output_file)
    print(f'Successfully processed metrics and wrote results to {output_file}')
