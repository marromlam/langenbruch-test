# test_crystall_ball
#
#


__all__ = []
__author__ = ["Marcos Romero"]
__email__ = ["mromerol@cern.ch"]


# Modules {{{

import os
import ipanema
import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import complot


ipanema.initialize('python', 1)
# prog = ipanema.compile(
# """
# #define USE_DOUBLE 1
# #include <exposed/kernels.ocl>
# """
# )

# }}}


# mass model {{{

def mass_model(mass, prob, fsig=1, fexp=0,
               a=0, b=0, c=0,
               norm=1, xLL=None, xUL=None):
  num = a * np.cos(b*mass+c)
  den = (a*(-np.sin(c + b*xLL) + np.sin(c + b*xUL)))/b
  # num = a * np.sin(b*mass+c)
  # den = (a*(np.cos(c + b*xLL) - np.cos(c + b*xUL)))/b
  return norm * ipanema.ristra.get(num/den)

# }}}


if __name__ == '__main__':

  # mass range
  mLL = 0
  mUL = 2

  # create a parameter set
  pars = ipanema.Parameters()
  pars.add({"name": "a", "value": 2,
            "free": False,
            "latex": r"\mu"})
  pars.add({"name": "b", "value": 4.3, #  "min":0, "max":1,
            "free": True,
            "latex": r"\sigma"})
  pars.add({"name": "c", "value": 3.2, "free": True,
            "latex": "a_l"})
  pars.add({"name": "xLL", "value": mLL,
            "latex": "m_l", "free": False})
  pars.add({"name": "xUL", "value": mUL,
            "latex": "m_u", "free": False})


  # generate dataset (if it is not there) {{{

  if not os.path.exists("test_cos.root"):
    print("Generating data")
    # lets generate a random histogram
    mass_h = np.linspace(mLL, mUL, 1000)
    mass_d = ipanema.ristra.allocate(mass_h)
    prob_d = 0*mass_h

    def lambda_model(mass):
      _mass = ipanema.ristra.allocate(mass).astype(np.float64)
      _prob = 0 * _mass
      return mass_model(mass=_mass, prob=_prob, **pars.valuesdict(), norm=1)

    pdfmax = np.max(ipanema.ristra.get(lambda_model(mass_h)))
    prob_h = ipanema.ristra.get(lambda_model(mass_h))
    print("pdfmax =", pdfmax)

    def generate_dataset(n, pdfmax=1):
      i = 0
      output = np.zeros(n)
      while i < n:
          V = (mUL-mLL) * np.random.rand() + mLL
          U = np.random.rand()
          pdf_value = lambda_model(np.float64([V]))[0]
          if U < 1/pdfmax * pdf_value:
              output[i] = V
              i = i + 1
      return output

    data_h = generate_dataset(int(1e6), 1.2*pdfmax)
    pandas_host = pd.DataFrame({"mass": data_h})
    with uproot.recreate("test_cos.root") as f:
      _branches = {}
      for k, v in pandas_host.items():
          if 'int' in v.dtype.name:
              _v = np.int32
          elif 'bool' in v.dtype.name:
              _v = np.int32
          else:
              _v = np.float64
          _branches[k] = _v
      mylist = list(dict.fromkeys(_branches.values()))
      f["DecayTree"] = uproot.newtree(_branches)
      f["DecayTree"].extend(pandas_host.to_dict(orient='list'))


    hdata = complot.hist(data_h, bins=60)
    plt.plot(hdata.bins, hdata.counts)
    plt.savefig("cos_histo.png")

  # }}}


  # fit data {{{

  print("Loading sample")
  sample = ipanema.Sample.from_root("test_cos.root")
  sample.allocate(mass="mass", prob="0*mass")

  # likelihood funtiion to optimize
  def fcn(params, data):
    p = params.valuesdict()
    prob = mass_model(mass=data.mass, prob=data.prob, **p)
    return -2.0 * np.log(prob)

  # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':sample}, method='nelder',
  #                        verbose=False, strategy=2)
  # print(f"Nelder = \n{res.params}")
  # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':sample}, method='bfgs',
  #                        verbose=False, strategy=2)
  # print(f"BFGS = \n{res.params}")
  res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':sample}, method='minos',
                         verbose=True, strategy=2)
  print(f"Minuit = \n{res}")
  fpars = ipanema.Parameters.clone(res.params)

  # }}}


  # plot {{{

  _p = fpars.valuesdict()
  fig, axplot, axpull = complot.axes_plotpull()
  hdata = complot.hist(ipanema.ristra.get(sample.mass), bins=100, density=False)
  axplot.errorbar(
      hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr, fmt=".k"
  )

  mass = ipanema.ristra.linspace(ipanema.ristra.min(sample.mass), ipanema.ristra.max(sample.mass), 200)
  signal = 0 * mass

  # plot signal: nbkg -> 0 and nexp -> 0
  for icolor, pspecie in enumerate(fpars.keys()):
    if pspecie.startswith('f'):
      _p = ipanema.Parameters.clone(fpars)
      for f in _p.keys():
            if f.startswith('f'):
              if f != pspecie:
                _p[f].set(value=0, min=-np.inf, max=np.inf)
              else:
                _p[f].set(value=fpars[pspecie].value, min=-np.inf, max=np.inf)

      _x, _y = ipanema.ristra.get(mass), ipanema.ristra.get(
          mass_model(mass, signal, **_p.valuesdict(), norm=hdata.norm)
      )
      _label = f"${fpars[pspecie].latex.split('f_')[-1]}$"
      axplot.plot(_x, _y, color=f"C{icolor+1}", label=_label)

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(fpars)
  x, y = ipanema.ristra.get(mass), ipanema.ristra.get(
      mass_model(mass, signal, **_p.valuesdict(), norm=hdata.norm)
  )
  axplot.plot(x, y, color="C0")
  pulls = complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr)
  axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)

  # label and save the plot
  axpull.set_xlabel(r"$m$ [MeV/$c^2$]")
  axpull.set_ylim(-6.5, 6.5)
  axpull.set_yticks([-5, 0, 5])
  axpull.hlines(+3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  axpull.hlines(-3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  axplot.set_ylabel(r"Candidates")
  axplot.legend(loc="upper right", prop={'size': 8})
  fig.savefig("test_cos.pdf")
  plt.close()

  # }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
