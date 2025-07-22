from bitflags import BitFlags
import json

class ErrorFlags(BitFlags):
    """
    Flags for different error types in the Kalman filter.
    """

    options = {
        #* Error flags
        0 : "TOO_FEW_MEASUREMENTS",         #* 0b0000001
        1 : "MOMENTUM_TOO_LOW",             #* 0b0000010
        2 : "OUTSIDE_REGION",               #* 0b0000100
        3 : "NEG_EIGENVALUES_COV_MATRIX",   #* 0b0001000
        4 : "DETECTOR_MISS",                #* 0b0010000
        5 : "ED_NEQUAL_MEASURED",           #* 0b0100000
        6 : "NOT_IN_ERROR_ELLIPSE",         #* 0b1000000
    }

    def filtering(self) -> bool:
        """
        Check if any filtering error flags are set.
        """
        
        return self.bit_0 | self.bit_1 | self.bit_2

    def list_active_flags(self):
        """
        List all active error flags.
        """
        return [flag for bit, flag in self.options.items() if getattr(self, f"bit_{bit}")]
    
    def to_dict(self):
        """
        Dictionary representation of active error flags.
        """
        return {name : True for bit, name in self.options.items() if getattr(self, f"bit_{bit}")}
    
    def to_json(self):
        """
        JSON representation of active error flags.
        """
        return json.dumps(self.to_dict(), indent=4)