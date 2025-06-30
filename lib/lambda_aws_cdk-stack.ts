import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';

export class LambdaAwsCdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================
    // EXISTING FINANCIAL ANALYSIS FUNCTION
    // ========================================
    const financialAnalysisFunc = new lambda.DockerImageFunction(this, "FinancialAnalysisFunc", {
      code: lambda.DockerImageCode.fromImageAsset("./image"),
      memorySize: 1024,
      timeout: cdk.Duration.seconds(10),
      architecture: lambda.Architecture.ARM_64,
      description: "Financial data analysis and scoring service",
      functionName: "financial-analysis-service",
    });
    
    const financialAnalysisFunctionUrl = financialAnalysisFunc.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedMethods: [lambda.HttpMethod.ALL],
        allowedHeaders: ["*"],
        allowedOrigins: ["*"],
      },
    });

    // ========================================
    // NEW FUNCTION OPTION 1: ANOTHER DOCKER FUNCTION
    // ========================================
    // If you want to create another Docker-based function from a different image
    /*
    const secondDockerFunc = new lambda.DockerImageFunction(this, "SecondDockerFunc", {
      code: lambda.DockerImageCode.fromImageAsset("./image2"), // Different image folder
      memorySize: 512,
      timeout: cdk.Duration.seconds(30),
      architecture: lambda.Architecture.ARM_64,
      description: "Second service function",
      functionName: "second-service",
    });

    const secondFunctionUrl = secondDockerFunc.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedMethods: [lambda.HttpMethod.ALL],
        allowedHeaders: ["*"],
        allowedOrigins: ["*"],
      },
    });
    */

    // ========================================
    // NEW FUNCTION OPTION 2: ZIP-BASED PYTHON FUNCTION
    // ========================================
    // If you want a simpler Python function without Docker
    const simplePythonFunc = new lambda.Function(this, "SimplePythonFunc", {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: "index.lambda_handler",
      code: lambda.Code.fromInline(`
import json
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """
    Simple example function - you can replace this with your actual logic
    """
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        },
        'body': json.dumps({
            'message': 'Hello from second Lambda function!',
            'timestamp': datetime.now().isoformat(),
            'event': event
        })
    }
      `),
      timeout: cdk.Duration.seconds(10),
      memorySize: 128,
      description: "Simple Python function example",
      functionName: "simple-python-service",
    });

    const simplePythonFunctionUrl = simplePythonFunc.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedMethods: [lambda.HttpMethod.ALL],
        allowedHeaders: ["*"],
        allowedOrigins: ["*"],
      },
    });

    // ========================================
    // NEW FUNCTION OPTION 3: ZIP-BASED FUNCTION FROM FILE
    // ========================================
    // If you want to deploy code from a separate directory
    /*
    const fileBasedFunc = new lambda.Function(this, "FileBasedFunc", {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: "main.lambda_handler",
      code: lambda.Code.fromAsset("./lambda-functions/my-function"), // Local directory
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      description: "Function deployed from local files",
      functionName: "file-based-service",
    });
    */

    // ========================================
    // OUTPUTS
    // ========================================
    new cdk.CfnOutput(this, 'FinancialAnalysisFunctionUrl', {
      value: financialAnalysisFunctionUrl.url,
      description: 'URL for the Financial Analysis Function',
    });

    new cdk.CfnOutput(this, 'SimplePythonFunctionUrl', {
      value: simplePythonFunctionUrl.url,
      description: 'URL for the Simple Python Function',
    });
  }
}
