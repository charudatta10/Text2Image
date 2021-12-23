import logging
import numpy as np

from visdom import Visdom

class Visualizer():
    """
    Visualizes the provided plots on a Visdom server for live updates.
    """

    def __init__(self, environment : str):
        """
        Initialize the Visdom server and associated environment.

        Args:
            environment (str): Defines the current Visdom environment 
                               (prevent plotting clashes).
        """
        # Define a visualization specific logger.
        self._logger = logging.getLogger(F"Visdom Visualization Logger")
        
        # Construct visdom environment and plots.
        self._visualizer = Visdom()
        self._environment = environment
        self._plots = {}

    def __update_plot(self, *,
                      plot_name : str, 
                      component_name : str, 
                      x : np.ndarray, 
                      y : np.ndarray):
        """
        Update the specified plot with the provided 
        x and y value using the append operation.

        Args:
            plot_name (str): Name of the plot to be updated.
            component_name (str): Name of the component to be updated.
            x (np.ndarray): X point to be updated.
            y (np.ndarray): Y point to be updated.

        Raises:
            AttributeError: Incorrect plot name (plot name doesn't exist).
            update_plot_exception: An issue occured while attempting to update plot.
        """
        try:
            # Ensure that a plot of the specified name already exists.
            if plot_name not in self._plots:
                raise AttributeError(
                    F"ERROR: A plot with name {plot_name} does not exist!")

            # Update the specified plot with the provided data.
            self._visualizer.line(
                X = np.array(x),
                Y = np.array(y),
                env = self._environment,
                win = self._plots[plot_name],
                name = component_name,
                update = "append")

        except Exception as update_plot_exception:
            self._logger.error(
                F"ERROR: Issue occured while attempting to update plot : {plot_name}.")
            raise update_plot_exception

    def __create_plot(self, *, 
                      plot_name : str, 
                      component_name : str, 
                      title : str, 
                      xlabel : str, 
                      ylabel : str, 
                      x : np.ndarray, 
                      y : np.ndarray):
        """
        Constructs a plot with the specified name, component, 
        title and labels.

        Args:
            plot_name (str): Name associated with the plot.
            component_name (str): Name of the current component.
            title (str): Title of the plot to be created.
            xlabel (str): X-label of the plot to be created.
            ylabel (str): Y-label of the plot to be created.
            x (np.ndarray): X point to be plotted.
            y (np.ndarray): Y point to be plotted.

        Raises:
            AttributeError: Incorrect plot name (plot name already exists).
            create_plot_exception: An issue occured while attempting to 
                                   visualize training performance
        """
        try:
            # Ensure that a plot of the specified name does not already exist.
            if plot_name in self._plots:
                raise AttributeError(
                    F"ERROR: A plot with name {plot_name} already exists!")

            # Plot the provided data on the visdom dashboard.
            self._plots[plot_name] = self._visualizer.line(
                X = np.array(x),
                Y = np.array(y),
                env = self._environment,
                name = component_name,
                opts = dict(
                    title = title,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    showlegend = True
                )
            )

        except Exception as create_plot_exception:
            self._logger.error(
                "ERROR: Issue occured while attempting to visualize training performance.")
            raise create_plot_exception

    def plot(self, *, 
             plot_name : str,
             component_name : str,
             title : str = None,
             xlabel : str = None,
             ylabel : str = None,
             x : np.ndarray,
             y : np.ndarray):
        """
        Constructs or updates a plot with the specified name, component, 
        title and labels.

        Args:
            plot_name (str): Name associated with the plot.
            component_name (str): Name of the current component.
            title (str): Title of the plot to be created.
            xlabel (str): X-label of the plot to be created.
            ylabel (str): Y-label of the plot to be created.
            x (np.ndarray): X point to be plotted.
            y (np.ndarray): Y point to be plotted.

        Raises:
            create_plot_exception: An issue occured while attempting to 
                                   visualize training performance
        """
        try:
            # If the plot exists, update the plot, otherwise create a new plot.
            if plot_name in self._plots:
                self.__update_plot(
                    plot_name = plot_name,
                    component_name = component_name,
                    x = x,
                    y = y)

            else:
                self.__create_plot(
                    plot_name = plot_name,
                    component_name = component_name,
                    title = title,
                    xlabel = xlabel,
                    ylabel = ylabel,
                    x = x,
                    y = y)

        except Exception as plotting_exception:
            self._logger.error(
                F"ERROR: Issue occured while attempting to plot on {plot_name}.")
            raise plotting_exception