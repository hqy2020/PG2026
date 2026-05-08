% ============================================================
% egPublStyle.cls — Minimal CGF-style class for PG2026
% ============================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{egPublStyle}[2024/01/01 CGF style for PG2026]

\LoadClass[10pt,twoside]{article}

\RequirePackage{geometry}
\geometry{left=2cm,right=2cm,top=2.5cm,bottom=2.5cm,columnsep=8mm}

\RequirePackage[font=small,labelfont=bf]{caption}
\RequirePackage{fancyhdr}
\RequirePackage{times}
\RequirePackage{graphicx}
\RequirePackage{amssymb,amsmath}
\RequirePackage{booktabs}
\RequirePackage{microtype}

% Header/Footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting}
\fancyhead[R]{\small Q. Hu et al.}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Title
\renewcommand{\maketitle}{
  \twocolumn[
    \begin{center}
      {\LARGE\bfseries \@title \par}
      \vspace{8pt}
      {\large \@author \par}
      \vspace{4pt}
      {\small \@date \par}
      \vspace{12pt}
    \end{center}
  ]
}

\newcommand{\conference}[1]{\def\@conference{#1}}
\AtBeginDocument{\ifx\@conference\undefined\else\twocolumn[\begin{center}\small\@conference\end{center}]\fi}

% Section formatting
\RequirePackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\small\bfseries}{\thesubsubsection}{1em}{}
\titlespacing*{\section}{0pt}{12pt}{4pt}
\titlespacing*{\subsection}{0pt}{8pt}{2pt}
\titlespacing*{\subsubsection}{0pt}{6pt}{2pt}

% Bibliography
\RequirePackage[numbers,sort&compress]{natbib}
\bibliographystyle{abbrv}
