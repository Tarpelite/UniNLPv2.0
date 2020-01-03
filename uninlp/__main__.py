# coding: utf8
def main():
    import sys
    if (len(sys.argv) < 4 or len(sys.argv) > 6) or sys.argv[1] not in ["bert", "gpt", "transfo_xl", "gpt2", "xlnet", "xlm"]:
        print(
        "This command line utility let you convert original (author released) model checkpoint to pytorch.\n"
        "It should be used as one of: \n"
        ">> transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT, \n"
        ">> transformers gpt OPENAI_GPT_CHECKPOINT_FOLDER_PATH PYTORCH_DUMP_OUTPUT [OPENAI_GPT_CONFIG], \n"
        ">> transformers transfo_xl TF_CHECKPOINT_OR_DATASET PYTORCH_DUMP_OUTPUT [TF_CONFIG] or \n"
        ">> transformers gpt2 TF_CHECKPOINT PYTORCH_DUMP_OUTPUT [GPT2_CONFIG] or \n"
        ">> transformers xlnet TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT [FINETUNING_TASK_NAME] or \n"
        ">> transformers xlm XLM_CHECKPOINT_PATH PYTORCH_DUMP_OUTPUT")
    else:
        if sys.argv[1] == "bert":
            try:
                from .convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
            except ImportError:
                print("transformers can only be used from the commandline to convert TensorFlow models in PyTorch, "
                    "In that case, it requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
                raise

            if len(sys.argv) != 5:
                # pylint: disable=line-too-long
                print("Should be used as `transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT`")
            else:
                PYTORCH_DUMP_OUTPUT = sys.argv.pop()
                TF_CONFIG = sys.argv.pop()
                TF_CHECKPOINT = sys.argv.pop()
                convert_tf_checkpoint_to_pytorch(TF_CHECKPOINT, TF_CONFIG, PYTORCH_DUMP_OUTPUT)

if __name__ == '__main__':
    main()
