class Log:
    _instance = None
    _initialized = False
    
    LEVELS = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3
    }

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "RESET": "\033[0m"      # Reset
    }
    
    def __new__(cls, level="DEBUG", for_e2e=False):
        if cls._instance is None:
            cls._instance = super(Log, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, level="DEBUG", for_e2e=False):
        if not self._initialized:
            if level not in self.LEVELS:
                raise ValueError(f"Invalid level: {level}. Use: {list(self.LEVELS.keys())}")
            
            self.current_level = level
            self.current_priority = self.LEVELS[level]
            self.for_e2e = for_e2e
            Log._initialized = True
    
    def log(self, message, level="INFO"):
        if level not in self.LEVELS:
            raise ValueError(f"Invalid level: {level}. Use: {list(self.LEVELS.keys())}")
        
        message_priority = self.LEVELS[level]
        
        if message_priority >= self.current_priority:
            if self.for_e2e:
                Level = level[0] + level[1:].lower()
                print(f"{Level} | {message}\n")
            else:
                colored_level = f"{self.COLORS[level]}{level}{self.COLORS['RESET']}"
                print(f"[{colored_level}] {message}\n")
    
    def debug(self, message):
        self.log(message, "DEBUG")
    
    def info(self, message):
        self.log(message, "INFO")
    
    def warning(self, message):
        self.log(message, "WARNING")
    
    def error(self, message):
        self.log(message, "ERROR")
    
    def set_level(self, level):
        if level not in self.LEVELS:
            raise ValueError(f"Invalid level: {level}. Use: {list(self.LEVELS.keys())}")
        
        self.current_level = level
        self.current_priority = self.LEVELS[level]
