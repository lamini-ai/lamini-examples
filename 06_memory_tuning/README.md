<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Memory Tuning Llama 3.1

It's easy to finetune Llama 3.1 on the question & answer dataset we just created.

```bash
cd 06_memory_tuning
python3 train.py
```

# Monitoring your job

After you submit a training job, it is scheduled on the cluster. You can monitor the progress of the job by visiting the link provided in the output of the training script.

```bash
$ python3 train.py

Uploading data....
Upload to blob completed for data.
Data pairs uploaded to blob.

Your dataset id is: 0713f8cfa5746a0897079e7f249a048deb653cf7e849d6fc26f3d2dacc5722d0 . Consider using this in the future to train using the same data.
Eg: llm.train(dataset_id='0713f8cfa5746a0897079e7f249a048deb653cf7e849d6fc26f3d2dacc5722d0')
Training job submitted! Check status of job 6367 here: https://app.lamini.ai/train/6367
```

The page lets you monitor all of your jobs, view eval results, view loss curves, and logs.

![image](https://github.com/lamini-ai/lamini-earnings-sdk/assets/3401278/f7db9547-88d1-4983-8217-f21c3a3f3da0)
