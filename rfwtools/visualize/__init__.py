import seaborn as sns

# Define some standard marker sets and palettes
zone_markers = {'0L04': 'd', '1L07': 'D', '1L22': 'o', '1L23': 'v', '1L24': '^', '1L25': '<', '1L26': '>', '2L22': '8',
                '2L23': 's', '2L24': 'p', '2L25': '*', '2L26': 'X'}

cavity_markers = {'0': 'd', '1': 'D', '2': 'o', '3': 'v', '4': '^', '5': '<', '6': '>', '7': '8', '8': 's'}

fault_markers = {'Single Cav Turn off': 'd', 'Multi Cav turn off': 'D', 'E_Quench': 'o', 'Quench_3ms': 'v',
                 'Quench_100ms': '^', 'Microphonics': '<', 'Controls Fault': '>', 'Heat Riser Choke': '8',
                 'Unknown': 's'}
