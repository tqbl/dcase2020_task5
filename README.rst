Surrey CVSSP DCASE 2020 Task 5 System
=====================================

This is the source code for CVSSP's `DCASE 2020 Task 5`__ submission.

For more details about the system, consider reading the `technical
report`__ [1]_.

__ http://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context
__ http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Iqbal_38_t5.pdf


.. contents::


Requirements
------------

This software requires Python >=3.6. To install the dependencies, run::

    poetry install

or::

    pip install -r requirements.txt

DCASE 2020 Task 5 uses the SONYC-UST v2 dataset, which may be downloaded
`here`__. For convenience, a bash script is provided to download the
dataset automatically. The dependencies are bash, curl, and tar. Simply
run the following command from the root directory of the project::

    $ scripts/download_dataset.sh

This will download the dataset to a directory called ``_dataset/``. When
running the software, use the ``--dataset_path`` option (refer to the
`Usage`_ section) to specify the path of the dataset directory. This is
only necessary if the dataset path is different from the default.

__ https://zenodo.org/record/3873076


Quick Start
-----------

To run the experiments that were presented in the technical report,
there are several bash scripts available. Assuming the dataset has been
downloaded in ``_dataset/``, run these commands::

    $ scripts/run.sh
    $ scripts/evaluate.sh

Note that various files will be created in a directory called
``_workspace``, which itself is created in the working directory. Ensure
that enough hard disk space is available (a few GBs at most). To change
the path of the workspace directory, modify the configuration files in
`scripts/`__ and `scripts/evaluate.sh`__. The same applies if you have
downloaded the dataset in a different directory. More details about
configuring the software can be found in the next section.

__ scripts
__ scripts/evaluate.sh


Usage
-----

The general usage pattern is::

    python <script> [-f PATH] <args...> [options...]

The various options can also be specified in a configuration file. Using
the ``--config_file`` (or ``-f``) command-line option, the path of the
configuration file can be specified to the program. Options that are
passed in the command-line override those in the config file. See
`default.conf`__ for an example of a config file. It also includes
descriptions of each option. Note that this file is generally not
intended to be modified, with the exception being the paths.

In the following subsections, the various commands are described. Using
this program, the user is able to extract feature vectors, train the
network, compute predictions, and evaluate the predictions.

__ default.conf

Feature Extraction
^^^^^^^^^^^^^^^^^^

To extract feature vectors, run::

    python task5/extract.py <training/validation/test> [--dataset_dir DIR] [--extraction_dir DIR] [--sample_rate RATE] [--n_fft N] [--hop_length N] [--n_mels N] [--overwrite BOOL]

This extracts log-mel spectrograms and stores them in a HDF5 file.

Pseudo-labeling
^^^^^^^^^^^^^^^

To generate pseudo-labels, run::

    python task5/relabel.py [--dataset_dir DIR] [--minimum N] [--n_epochs N] [--validate BOOL] [--output_path PATH]

This will train a simple two-layer neural network to estimate labels for
the training set. To avoid confusion, we will refer to the training set
used to train the simple neural network as the pseudo-labeling training
set (PTS), while the training set of the SONYC-UST dataset will simply
be called the training set.

The ``--minimum`` option is used to set the minimum number of PTS
instances per class. In order to satisfy this requirement, instances
from the training set are sampled and included in the PTS. Setting this
option to zero will disable sampling. The drawback of not sampling is
that the PTS will under-represent several of the classes to the point
where some classes will have no examples.

The number of training epochs can be specified using ``--n_epochs``.

Use the ``--validate`` option to use a subset of the PTS for validation.

Use the ``--output_path`` option to specify the output path. By default,
the output path is ``metadata/pseudolabels.csv``, meaning the
`pseudolabels.csv`__ file that is already provided will be overwritten.

__ metadata/pseudolabels.csv

Training
^^^^^^^^

To train a model, run::

    python task5/train.py [--dataset_dir DIR] [--extraction_dir DIR] [--model_dir DIR] [--log_dir DIR] [--pseudolabel_path PATH] [--training_id ID] [--model MODEL] [--training_mask MASK] [--validation_mask MASK] [--seed N] [--batch_size N] [--n_epochs N] [--lr NUM] [--lr_decay NUM] [--lr_decay_rate N] [--use_stc BOOL] [--augment BOOL] [--overwrite BOOL]

The ``--model`` option accepts the following values:

* ``gcnn`` - Use the randomly-initialized GCNN model.
* ``qkcnn10`` - Use the pre-trained CNN10 (PANN) model.

The ``--training_id`` option is used to differentiate training runs, and
partially determines where the models are saved. When running multiple
trials, either use the ``--seed`` option to specify different random
seeds or set it to a negative number to disable setting the random seed.
Otherwise, the learned models will be identical across different trials.

Use the ``--pseudolabel_path`` option to specify where the pseudo-labels
are located. By default, this option is not specified, which will
disable the use of pseudo-labels for training.

Use the ``--use_stc`` option to enable (default) or disable the use of
spatiotemporal context (STC) features.

Prediction
^^^^^^^^^^

To compute predictions, run::

    python task5/predict.py <validation/test> [--dataset_dir DIR] [--extraction_dir DIR] [--model_dir DIR] [--log_dir DIR] [--prediction_dir DIR] [--training_id ID] [--use_stc BOOL] [--mask MASK] [--epochs EPOCHS] [--clean BOOL]

By default, it will average the predictions of the top three epochs
(based on the macro AUPRC metric). To change this behavior, use the
``--epochs`` option, which accepts either a list of epoch numbers or a
specification of the form ``metric:n``. The default value of this option
is ``val_auprc_macro:3``.

Evaluation
^^^^^^^^^^

To evaluate the predictions, we have integrated the code from the
`official baseline`__. Run::

    python task5/extern/evaluate_predictions.py [-h] <prediction_path> <annotation_path> <yaml_path>

__ https://github.com/sonyc-project/dcase2020task5-uststc-baseline/


Citing
------
If you wish to cite this work, please cite the following paper:

.. [1] \T. Iqbal, Y. Cao, M. D. Plumbley, and W. Wang, "Incorporating
       Auxiliary Data for Urban Sound Tagging," DCASE2020 Challenge,
       Tech. Rep., October 2020.

BibTeX::

    @techreport{Iqbal2020,
        author = {Iqbal, Turab and Cao, Yin and Plumbley, Mark D. and Wang, Wenwu},
        title = {Incorporating Auxiliary Data for Urban Sound Tagging},
        institution = {DCASE2020 Challenge},
        year = {2020},
        month = {October},
    }
