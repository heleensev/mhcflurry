#!/usr/bin/env python
#
# Copyright (c) 2015. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join, exists
from os import makedirs
from argparse import ArgumentParser

from test_data import load_test_data

parser = ArgumentParser()


parser.add_argument(
    "--test-data-input-dirs",
    nargs='*',
    type=str,
    help="Multiple directories from other predictors",
    required=True)

parser.add_argument(
    "--test-data-input-sep",
    default="\s+",
    help="Separator to use for loading test data CSV/TSV files",
    type=str)

parser.add_argument(
    "--test-data-output-dir",
    help="Save combined test datasets to this directory",
    required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    dataframes, predictor_names = load_test_data(args.test_data_input_dirs)
    if not exists(args.test_data_output_dir):
        makedirs(args.test_data_output_dir)

    print("Loaded test data:")
    for (allele, df) in dataframes.items():
        df.index.name = "sequence"
        print("%s: %d results" % (allele, len(df)))
        filename = "blind-%s.csv" % allele
        filepath = join(args.test_data_output_dir, filename)
        df.to_csv(filepath)

    assert False
    """
    combined_df = evaluate_model_configs(
        configs=configs,
        results_filename=args.output,
        train_fn=lambda config: evaluate_model_config_train_vs_test(
            config,
            training_allele_datasets=training_datasets,
            testing_allele_datasets=testing_datasets,
            min_samples_per_allele=5))
    """
