import numpy
import vtk

# TODO!
def getData():
    """Retrieve the points as an Numpy array. 

    Note the numpy.save and numpy.load functions for implementing this.
    These functions are favored over pickling.
    """
    return numpy.zeros((1,1,1))


def numpy_to_vtk(matrix, data_spacing=None, data_extent=None):
    """Transform a Numpy matrix to a VTK data source.
    
    Arguments:
        - matrix: The Numpy data
        - data_spacing: (x_spacing, y_spacing, z_spacing) or None
        - data_extent: (xmin, xmax, ymin, ymax, zmin, zmax) or None
    """
    if data_spacing is None:
        data_spacing = (1, 1, 1)
    if data_extent is None:
        data_extent = (0, matrix.shape[0],
                       0, matrix.shape[1],
                       0, matrix.shape[2])
    data = matrix.tostring()
    data_reader = vtk.vtkImageImport()
    data_reader.CopyImportVoidPointer(data, len(data))
    data_reader.SetDataScalarTypeToUnsignedChar()
    data_reader.SetNumberOfScalarComponents(1)
    data_reader.SetDataSpacing(*data_spacing)
    data_reader.SetDataExtent(*data_extent)
    data_reader.SetWholeExtent(*data_extent)
    return data_reader


if __name__ == "__main__":
    np_data = getData()
    vtk_data = numpy_to_vtk(np_data)

    contour = vtk.vtkMarchingCubes()
    contour.SetInputConnection(vtk_data.GetOutputPort())
    contour.ComputeNormalsOn()
    contour.SetValue(0, 7)

    geometry = vtk.vtkPolyDataMapper()
    geometry.SetInputConnection(contour.GetOutputPort())
    geometry.ScalarVisibilityOff()

    #actor = vtk.vtkLODActor()
    actor = vtk.vtkOpenGLActor()
    #actor.SetNumberOfCloudPoints(1000000)
    actor.SetMapper(geometry)
    actor.GetProperty().SetColor(0.333, 0.333, 0)#colors.green_to_white(0, 255))
    actor.GetProperty().SetOpacity(0.3)#colors.default_opacity(0, 255, 0.5))

    renderer = vtk.vtkOpenGLRenderer()
    render_window = vtk.vtkRenderWindow()

    renderer.AddActor(actor)
    renderer.SetBackground(0.32157, 0.34118, 0.43137)
    render_window.SetSize(500, 500)

    render_window.AddRenderer(renderer)
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_window)
    def exit_check(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    render_window.AddObserver("AbortCheckEvent", exit_check)

    render_interactor.Initialize()
    render_window.Render()
    render_interactor.Start()
