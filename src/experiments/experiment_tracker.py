from multiprocessing import Lock, Manager, Value

class ExperimentTracker:
    def __init__(self):
        self.comparison_metric = 'mse'
        self.best_value = Value("d", float("inf"))
        self.best_aunl = Value("d", float("inf"))       
        self.best_ge = Value("d", float("inf"))   
        self.best_weights = None

        manager = Manager()
        self.candidates = manager.list() 
        self.pruned = manager.list() 
        self.log_queue = manager.Queue()        
        self.budget = Value("i", 100)  # Shared budget variable

        self.weights_lock = Lock()
