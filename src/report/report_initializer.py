import src.utils.os_utils as os_utils
from src.config.file_constants import OUTPUT_DIR
import src.report.report_data_filter as filter
import streamlit as st
import src.utils.plots as plots
import numpy as np


class ReportInitializer:
    """
    Generates streamlit report.
    Summarizes some aspects of the experiment performance.
    """

    def __init__(self):
        self.load_data()
        self.experiment_comparison = False

    def load_data(self):
        self.data = os_utils.load_experiment_data(OUTPUT_DIR)
        self.experiments = os_utils.get_experiment_ids(OUTPUT_DIR)

    def get_download_button(self, fig, plot, experiment, col):
        # Convert to SVG for download
        svg_data = plots.get_svg_from_figure(fig)

        # Add download button
        st.download_button(
            key=f"download_{col}",
            label="📥 Download SVG",
            data=svg_data,
            file_name=f"{experiment}_{plot}.svg",
            mime="image/svg+xml",
        )

    def get_plot_data(self, plot, experiment_id):
        if plot == "whisker":
            return filter.get_whisker_plot_data(self.data[experiment_id])
        elif plot == "tabular":
            return filter.get_tabular_data(self.data[experiment_id])
        elif plot == "tabluar_experiment":
            return filter.get_experiment_parameters(self.data[experiment_id])
        elif plot == "tabular_hpo":
            return filter.get_hpo_algorithm_params(self.data[experiment_id])
        elif plot == "tabular_summary":
            return filter.get_experiment_summary(self.data[experiment_id])
        elif plot == "epoch_frequency":
            return filter.get_epoch_frequency(self.data[experiment_id])

    def get_experiment_data(self, experiment):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.text("Experiment parameters")
            st.table(self.get_plot_data("tabluar_experiment", experiment))
        with col2:
            st.text("HPO parameters")
            st.table(self.get_plot_data("tabular_hpo", experiment))

    def get_tabular_content(self, experiment, col):
        col1, _ = st.columns([1, 5])
        with col1:
            # Create a unique key by including both the experiment and option
            select_model = st.selectbox(
                "Select model",
                ("Best", "Mode", "Median", "Mean", "Worst"),
                key=f"select_model_{str(experiment)}_{col}",  # Make key unique by adding 'option'
                label_visibility="hidden",
            )

        df = self.get_plot_data("tabular", experiment)[select_model]
        df.index = filter.get_epochs(self.data[experiment])

        st.dataframe(df.T, width=1000)

        self.get_experiment_data(experiment)

    def get_summary_content(self, experiment):
        st.text("Summary")
        st.table(self.get_plot_data("tabular_summary", experiment))
        self.get_experiment_data(experiment)

    def get_aunl_content(self, experiment, col):
        data_cleaned = self.data[experiment][~self.data[experiment].isin([np.inf, -np.inf]).any(axis=1)]
        best_id = filter.get_model_based_on_performance(
            data_cleaned, "aunl_val", True
        )
        worst_id = filter.get_model_based_on_performance(
            data_cleaned, "aunl_val", False
        )

        epochs = filter.get_epochs(data_cleaned)
        models = [
            filter.get_normalized_loss(data_cleaned, "val_loss", best_id),
            filter.get_aggregated_loss(data_cleaned, "val_loss", "mode"),
            filter.get_aggregated_loss(data_cleaned, "val_loss", "median"),
            filter.get_aggregated_loss(data_cleaned, "val_loss", "mean"),
            filter.get_normalized_loss(data_cleaned, "val_loss", worst_id),
        ]

        fig = plots.plot_model_comparison(
            models[0],  # Best model
            models[1:],  # Other models (mode, median, mean, worst)
            epochs,
            [
                "Mode Model",
                "Median Model",
                "Mean Model",
                "Worst Model",
            ],
        )

        if self.experiment_comparison:
            st.pyplot(fig)
            self.get_download_button(fig, "aunl", experiment, col)
            self.get_experiment_data(experiment)
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.pyplot(fig)
                self.get_download_button(fig, "aunl", experiment, col)
            with col2:
                self.get_experiment_data(experiment)

    def get_whisker_content(self, experiment, col):
        metrics = self.get_plot_data("whisker", experiment)
        epochs = filter.get_epochs(self.data[experiment])
        fig = plots.plot_whisker_plots(metrics, epochs)

        st.pyplot(fig)
        self.get_download_button(fig, "whisker", experiment, col)

        self.get_experiment_data(experiment)

    def get_line_plot(self, experiment, col):
        models = filter.get_tabular_data(self.data[experiment])
        epochs = filter.get_epochs(self.data[experiment])

        fig = plots.plot_metric_comparison(
            models["Best"],
            models["Mode"],
            models["Median"],
            models["Mean"],
            models["Worst"],
            epochs,
            ["mae", "mse", "rmse", "smape"],
        )

        if self.experiment_comparison:
            st.pyplot(fig)
            self.get_download_button(fig, "line", experiment, col)
            self.get_experiment_data(experiment)
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.pyplot(fig)
                self.get_download_button(fig, "line", experiment, col)
            with col2:
                self.get_experiment_data(experiment)

    def get_interactive_content(self, experiment, col):
        data = self.data[experiment][~self.data[experiment].isin([np.inf, -np.inf]).any(axis=1)]

        max_value = len(data["model_id"].unique()) - 1
        models_ids = data["model_id"].unique()

        col1, col2 = st.columns([2, 1])
        with col2:
            col3, col4 = st.columns([1, 1])
            with col3:
                model_index1 = st.number_input(
                    "Model 1 index",
                    key=f"model_1",
                    min_value=0,
                    max_value=max_value,
                    step=1,
                    format="%d",
                )

                id1 = models_ids[model_index1]
                st.table(filter.extract_model_params(data, id1))

            with col4:
                model_index2 = st.number_input(
                    "Model 2 index",
                    key=f"model_2",
                    min_value=0,
                    value=1,
                    max_value=max_value,
                    step=1,
                    format="%d",
                )

                id2 = models_ids[model_index2]
                st.table(filter.extract_model_params(data, id2))

        with col1:
            epochs = filter.get_epochs(data)
            model1 = filter.get_normalized_loss(
                data, "val_loss", models_ids[model_index1]
            )
            model2 = filter.get_normalized_loss(
                data, "val_loss", models_ids[model_index2]
            )

            fig = plots.plot_aunl_comparison(epochs, model1, model2, max(epochs))
            st.pyplot(fig)
            self.get_download_button(fig, "interactive", experiment, col)

    def get_epoch_frequency_content(self, experiment, col):
        fig = plots.plot_epoch_frequency(
            self.get_plot_data("epoch_frequency", experiment)
        )

        if self.experiment_comparison:
            st.pyplot(fig)
            self.get_download_button(fig, "epoch_frequency", experiment, col)
            self.get_experiment_data(experiment)
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.pyplot(fig)
                self.get_download_button(fig, "epoch_frequency", experiment, col)
            with col2:
                self.get_experiment_data(experiment)

    def get_content(self, option, experiment, col):
        if option == "Summary":
            self.get_summary_content(experiment)
        if option == "Tabular":
            self.get_tabular_content(experiment, col)
        if option == "AUNL":
            self.get_aunl_content(experiment, col)
        if option == "Whisker":
            self.get_whisker_content(experiment, col)
        if option == "Line":
            self.get_line_plot(experiment, col)
        if option == "Interactive":
            self.get_interactive_content(experiment, col)
        if option == "Epoch Frequency":
            self.get_epoch_frequency_content(experiment, col)

    def start_web(self):
        st.set_page_config(layout="wide")
        st.markdown("""
            <style>
            .stDataFrame {
                overflow-x: auto;
                display: block;
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
        # Using "with" notation
        comparisons = [
            "Summary",
            "Tabular",
            "AUNL",
            "Whisker",
            "Line",
            "Epoch Frequency",
        ]

        with st.sidebar:
            st.header("Comapre")
            select_option_level = st.selectbox(
                "Comparison entities",
                ("Models", "Experiments"),
                label_visibility="hidden",
            )

            selected_option_experiment1 = st.selectbox(
                "Select experiment 1", self.experiments
            )

            if select_option_level == "Experiments":
                selected_option_experiment2 = st.selectbox(
                    "Select experiment 2", self.experiments
                )
                self.experiment_comparison = True

            if select_option_level == "Models":
                comparisons.append("Interactive")

            st.header("Comparison method")
            selected_option_comparison = st.selectbox(
                "Comparison method",
                comparisons,
                label_visibility="hidden",
            )

        st.header(selected_option_comparison, divider="gray")

        if select_option_level == "Experiments":
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Experiment 1")
                self.get_content(
                    selected_option_comparison, selected_option_experiment1, "1"
                )
            with col2:
                st.subheader("Experiment 2")
                self.get_content(
                    selected_option_comparison, selected_option_experiment2, "2"
                )

        else:
            st.subheader("Experiment 1")
            self.get_content(
                selected_option_comparison, selected_option_experiment1, "1"
            )
