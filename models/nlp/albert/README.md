# ALBERT (A Lite BERT)

TensorFlow 2.1 implementation of pretraining and finetuning scripts for ALBERT, a state-of-the-art language model.

The original paper: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

### Overview

Language models help AWS customers to improve search results, text classification, question answering, and customer service routing. BERT and its successive improvements are incredibly powerful, yet complex to pretrain. Here we demonstrate how to train a faster, smaller, more accurate BERT-based model called [ALBERT](https://arxiv.org/abs/1909.11942) on Amazon SageMaker with the FSx filesystem and TensorFlow 2 models from [huggingface/transformers](https://github.com/huggingface/transformers). We can pretrain ALBERT for a fraction of the cost of BERT and achieve better accuracy on SQuAD.

![SageMaker -> EC2 -> FSx infrastructure diagram](https://user-images.githubusercontent.com/4564897/81020280-b207a100-8e25-11ea-8b57-38f0a09a7fb2.png
)

### How To Launch Training

1. Create an FSx volume.

2. Download the datasets onto FSx. The simplest way to start is with English Wikipedia.

3. Create an Amazon Elastic Container Registry (ECR) repository. Then build a Docker image from `docker/ngc_sagemaker.Dockerfile` and push it to ECR.

```bash
export ACCOUNT_ID=
export REPO=
export IMAGE=${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${REPO}:ngc_tf210_sagemaker
docker build -t ${IMAGE} -f docker/ngc_sagemaker.Dockerfile .
$(aws ecr get-login --no-include-email)
docker push ${IMAGE}
```

4. Define environment variables to point to the FSx volume. For a list, use a comma-separated string.

```bash
export SAGEMAKER_ROLE=arn:aws:iam::${ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20200101T123
export SAGEMAKER_IMAGE_NAME=${IMAGE}
export SAGEMAKER_FSX_ID=fs-123
export SAGEMAKER_SUBNET_IDS=subnet-123
export SAGEMAKER_SECURITY_GROUP_IDS=sg-123,sg-456
```

5. Launch the SageMaker job.

```bash
# Add the main folder to your PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/deep-learning-models/models/nlp

python launch_sagemaker.py \
    --source_dir=. \
    --entry_point=run_pretraining.py \
    --sm_job_name=albert-pretrain \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --load_from=scratch \
    --model_type=albert \
    --model_size=base \
    --batch_size=32 \
    --gradient_accumulation_steps=2 \
    --warmup_steps=3125 \
    --total_steps=125000 \
    --learning_rate=0.00176 \
    --optimizer=lamb \
    --log_frequency=10 \
    --name=myfirstjob
```

6. Launch a SageMaker finetuning job.

```bash
python launch_sagemaker.py \
    --source_dir=. \
    --entry_point=run_squad.py \
    --sm_job_name=albert-squad \
    --instance_type=ml.p3dn.24xlarge \
    --instance_count=1 \
    --load_from=scratch \
    --model_type=albert \
    --model_size=base \
    --batch_size=6 \
    --total_steps=8144 \
    --warmup_steps=814 \
    --learning_rate=3e-5 \
    --task_name=squadv2
```

7. Enter the Docker container to debug and edit code.

```bash
docker run -it -v=/fsx:/fsx --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm ${IMAGE} /bin/bash
```

<!-- ### Training results

These will be posted shortly. -->