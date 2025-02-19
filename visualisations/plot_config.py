import matplotlib.pyplot as plt

# Ensure text is converted to paths
plt.rcParams["svg.fonttype"] = "path"  # Converts text to paths
plt.rcParams["text.usetex"] = False    # Use mathtext instead of full LaTeX
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # Use a more Illustrator-friendly font
plt.rcParams["font.family"] = "serif"

plt.rcParams["font.size"] = 8  # General font size
plt.rcParams["axes.labelsize"] = 8  # Axis label size
plt.rcParams["xtick.labelsize"] = 8  # Tick labels
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["font.family"] = "serif"  # Use a journal-friendly font

plt.rcParams["text.usetex"] = False
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# plt.rcParams["text.usetex"] = True
# plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams["axes.linewidth"] = 1.5  # Axis border thickness
plt.rcParams["lines.linewidth"] = 1.5  # Line thickness
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
