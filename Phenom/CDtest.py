#python3
 
import time
import PyPhenom as ppi
 
 
def setSpot(phenom, imageSize, position):
    mode = ppi.SemViewingMode(ppi.ScanMode.Spot, ppi.ScanParams(16, 16, 1, ppi.DetectorMode.All, False, 0, ppi.Position((position[0] + 0.5) / imageSize[0] - 0.5, (position[1] + 0.5) / imageSize[1] - 0.5)))
    phenom.SetSemViewingMode(mode)
 
def writeSpectrum(spectrum, filename):
    msa = ppi.Spectroscopy.MsaData()
    msa.Spectroscopy = spectrum
    ppi.Spectroscopy.WriteMsaFile(msa, filename)
 
 
phenom = ppi.Phenom('Simulator', '', '')
dpp = phenom.Spectrometer
 
settings = ppi.LoadEidSettings()
dpp.ApplySettings(settings.map)
 
size = (10, 10)
for y in range(0, size[1]):
    for x in range(0, size[0]): 
        setSpot(phenom, size, (x, y))
        dpp.ClearSpectrum()
        dpp.Start()
        time.sleep(3)
        dpp.Stop()
        writeSpectrum(dpp.GetSpectrum(), 'Spectrum_{0:03d}-{1:03d}.msa'.format(x, y))