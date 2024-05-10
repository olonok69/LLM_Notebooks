# What is APP Engine
App Engine is a service in GCP that provides end-to-end application management. It allows you to deploy and scale your applications using pre-configured or custom runtimes. You can write code in any language and use various features such as automatic load balancing, auto scaling, managed platform updates, and application health monitoring. App Engine is a top-level container that includes the service, version, and instance resources that make up your app


#  Create an App Engine app within the current Google Cloud Project.

gcloud app create

# Deploy app
gcloud app deploy 

# Browse app
gcloud app browse

# Check logs
gcloud app logs tail -s default



# Links
### An overview of App Engine 
https://cloud.google.com/appengine/docs/an-overview-of-app-engine
### App Engine documentation
https://cloud.google.com/appengine/docs/language-landing
### About App Engine standard environment
https://cloud.google.com/appengine/docs/standard-environment
### App Engine flexible environment
https://cloud.google.com/appengine/docs/flexible
### Setting up your Google Cloud project for App Engine
https://cloud.google.com/appengine/docs/flexible/managing-projects-apps-billing
### About Custom runtimes
https://cloud.google.com/appengine/docs/flexible/custom-runtimes/about-custom-runtimes
### app.yaml
https://cloud.google.com/appengine/docs/flexible/reference/app-yaml?tab=python#syntax
### Configuring your app with app.yaml
https://cloud.google.com/appengine/docs/flexible/python/configuring-your-app-with-app-yaml

# Logging
### docs
https://cloud.google.com/logging/docs
### library
https://pypi.org/project/google-cloud-logging/