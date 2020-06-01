ECR_REPO=$1
cd ~/SageMaker/deep-learning-models/models/vision/detection/docker
ALGO=frcnn-tutorial
docker build -t ${ECR_REPO}/${ALGO} -f Dockerfile.frcnn .

# login to ECR
REGION=$(aws configure get region)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
FULLNAME="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${ALGO}"
$(aws ecr get-login --region ${REGION} --no-include-email)

# push image to ECR
docker tag ${ECR_REPO}/${ALGO} ${FULLNAME}
docker push ${FULLNAME}
echo ${FULLNAME}