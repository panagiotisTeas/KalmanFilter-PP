from bitflags import BitFlags
import json

class ErrorFlags(BitFlags):
    """
    Flags for different error types in the Kalman filter.
    """

    options = {
        #* Error flags
        0 : "TOO_FEW_MEASUREMENTS",         #* 0b00000001
        1 : "MOMENTUM_TOO_LOW",             #* 0b00000010
        2 : "OUTSIDE_REGION",               #* 0b00000100
        3 : "NEG_EIGENVALUES_COV_MATRIX",   #* 0b00001000
        4 : "DETECTOR_MISS",                #* 0b00010000
        5 : "ED_NEQUAL_MEASURED",           #* 0b00100000
        6 : "NOT_IN_ERROR_ELLIPSE",         #* 0b01000000
        7 : "NAN",                          #* 0b10000000
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