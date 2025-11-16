How to run:

```bash
sbatch submit-llama3.sh
STAGE=2 sbatch deepspeed.sh 
STAGE=3 sbatch deepspeed.sh 
```