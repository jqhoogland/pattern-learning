# Delayed Generalization: Unifying Grokking and Double Descent

## Interpolating between grokking and double descent

## Set-up 


```shell
pip install -r requirements.txt
pip install -e .
```

## Running the sweeps

Replace `config.yml` in the following with the relevant config file:

```shell
wandb sweep --project grokking <config.yml>
```

This will initialize a sweep. 

To run the sweep, run the following command:

```shell
wandb agent <sweep_id>
```

where `<sweep_id>` is the id of the sweep you want to run. You can find the sweep id by running `wandb sweep ls`.

You can pass an optional `--count` flag to the `wandb agent` command to specify the number of runs you want to execute. If you don't pass this flag, the agent will run until all the runs in the sweep are complete (for a grid sweep).

On a multi-GPU machine, you can run multiple agents in parallel through the following:

```shell
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep_id> &
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep_id> &
...
```

## Toy Model

See the jupyter notebooks in `toy_models` for more instructions.

