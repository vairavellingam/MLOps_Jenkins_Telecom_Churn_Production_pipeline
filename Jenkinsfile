@Library('jenkins-shared') _

pipeline {
    agent any

    environment {
        DOCKER_REPO = "vel23/mlops-jenkins-churn-prediction"
    }

    stages {
        stage('Checkout') {
            steps {
                gitCheckout(
                    'https://github.com/vairavellingam/MLOps_Churn_CICD.git',
                    '*/main',
                    'github-token'
                )
            }
        }

        stage('Training Pipeline') {
            steps {
                sh '''
                    pip install -e .
                    python training_pipeline.py
                '''
            }
        }

        stage('Build & Push Image') {
            steps {
                dockerBuildAndPush(DOCKER_REPO, 'dockerhub-token')
            }
        }

        stage('Install Kubectl') {
            steps {
                installKubectl()
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                k8sDeploy('kubeconfig')
            }
        }
    }

    post {
        success {
            echo 'Churn MLOps pipeline deployed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs.'
        }
    }
}
