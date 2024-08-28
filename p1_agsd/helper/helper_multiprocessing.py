import gc

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)



class Helper_Multiprocessing:
    
    def __init__(self, all_processes: list[multiprocessing.Process], shots_at_a_time: int=1):
        
        self.all_processes = all_processes
        self.shots_at_a_time = shots_at_a_time
        
        self.completed_processes = 0
        self.running_processes = 0
        self.current_index = 0
        
        return
    
    
    def check_running_processes(self):
        
        self.running_processes = 0
        for process in self.all_processes:
            if process.is_alive():
                self.running_processes += 1
        
        gc.collect()
        
        return
    
    
    def run_next_process(self):
        
        self.all_processes[self.current_index].start()
        self.current_index += 1
        
        return
    
    
    def run_all_processes(self):
        
        i = 0
        while self.current_index < len(self.all_processes):
            self.check_running_processes()
            if self.running_processes < self.shots_at_a_time:
                self.run_next_process()
            
            i = (i+1) % 1000 
            if i == 0:
                print('\n\nRunning process {}, {}/{}.\n\n'.format(self.running_processes, self.current_index, len(self.all_processes)))
            
        return
    
    