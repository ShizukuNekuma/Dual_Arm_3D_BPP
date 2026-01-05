import omni.ui
from omni.kit.widget.viewport import ViewportWidget
from typing import Union


class StagePreviewWidget:
    def __init__(self, usd_context_name: str = '', camera_path: str = None, resolution: Union[tuple, str] = None, *ui_args ,**ui_kw_args):
        """StagePreviewWidget contructor
        Args:
            usd_context_name (str): The name of a UsdContext the Viewport will be viewing.
            camera_path (str): The path of the initial camera to render to.
            resolution (x, y): The resolution of the backing texture, or 'fill_frame' to match the widget's ui-size
            *ui_args, **ui_kw_args: Additional arguments to pass to the ViewportWidget's parent frame
        """
        # Put the Viewport in a ZStack so that a background rectangle can be added underneath
        self.__ui_container = omni.ui.ZStack()
        with self.__ui_container:
            # Add a background Rectangle that is black by default, but can change with a set_style 
            omni.ui.Rectangle(style_type_name_override='ViewportBackgroundColor', style={'ViewportBackgroundColor': {'background_color': 0xff000000}})

            # Create the ViewportWidget, forwarding all of the arguments to this constructor
            self.__vp_widget = ViewportWidget(usd_context_name=usd_context_name, camera_path=camera_path, resolution=resolution, *ui_args, **ui_kw_args)

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.__vp_widget:
            self.__vp_widget.destroy()
            self.__vp_widget = None
        if self.__ui_container:
            self.__ui_container.destroy()
            self.__ui_container = None

    @property
    def viewport_api(self):
        # Access to the underying ViewportAPI object to control renderer, resolution
        return self.__vp_widget.viewport_api

    def set_style(self, *args, **kwargs):
        # Give some styling access
        self.__ui_container.set_style(*args, **kwargs)