import threading

class ConversationController:
    def __init__(self):
        self.user_speaking = False
        self.ai_speaking = False
        self.interrupt = False
        self.lock = threading.Lock()
    
    def start_user(self):
        with self.lock:
            self.user_speaking = True
            if self.ai_speaking:
                self.interrupt = True
    
    def stop_user(self):
        with self.lock:
            self.user_speaking = False
    
    def start_ai(self):
        with self.lock:
            self.ai_speaking = True
            self.interrupt = False
    
    def stop_ai(self):
        with self.lock:
            self.ai_speaking = False
    
    def should_interrupt(self):
        with self.lock:
            return self.should_interrupt