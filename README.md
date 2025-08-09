## Usage

0. **Setup**

    ```shell
    export PYTHONPATH=`pwd`:$PYTHONPATH
    ```

1. **Training**

    ```shell
    python tools/train.py [--config-name config[.yaml]] [trainer.devices=4] \
        [+data_root=$DATA_ROOT] [+label_root=$LABEL_ROOT] [+depth_root=$DEPTH_ROOT]
    ```

    * Override the default config file with `--config-name`.
    * You can also override any value in the loaded config from the command line, refer to the following for more infomation.
        * https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
        * https://hydra.cc/docs/advanced/hydra-command-line-flags/
        * https://hydra.cc/docs/advanced/override_grammar/basic/

2. **Testing**

    Generate the outputs for submission on the evaluation server:

    ```shell
    python tools/test.py [+ckpt_path=...]
    ```

3. **Visualization**

    1. Generating outputs

        ```shell
        python tools/generate_outputs.py [+ckpt_path=...]
        ```

    2. Visualization

        ```shell
        python tools/visualize.py [+path=...]
        ```