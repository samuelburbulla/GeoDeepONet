
# Dirichlet boundary condition for unit cube
class UnitCubeDirichletBC:
    def __init__(self, value_dict=None):
        assert all(x in ["left", "right", "bottom", "top"]
                   for x in value_dict.keys())
        self.value_dict = value_dict

    @staticmethod
    def _where(x):
        if x[0] < 1e-6:
            return "left"
        if x[0] > 1 - 1e-6:
            return "right"
        if x[1] < 1e-6:
            return "bottom"
        if x[1] > 1 - 1e-6:
            return "top"
        return None

    def is_dirichlet(self, x):
        where = self._where(x)
        return where in self.value_dict.keys()

    def value(self, x):
        where = self._where(x)
        value = self.value_dict[where]
        if callable(value):
            return value(x)
        else:
            return value
