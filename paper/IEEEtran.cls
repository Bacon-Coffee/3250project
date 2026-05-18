% =============================================================================
% IEEEtran.cls -- Minimal compatibility shim
% -----------------------------------------------------------------------------
% This is NOT the official IEEEtran.cls by Michael Shell.  The official class
% is distributed via CTAN as the `IEEEtran' package.  Because that class is
% not installed in the local TeX Live build, this shim re-implements just
% enough of the IEEEtran public interface (\IEEEoverridecommandlockouts,
% \IEEEauthorblockN, \IEEEauthorblockA, IEEEkeywords environment, the
% `conference' option) to typeset a two-column 10pt letter-paper paper with
% IEEE-like section numbering and margins.  Drop in the official IEEEtran.cls
% to get the canonical look.
% =============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2026/05/18 v0-compat IEEEtran shim for course use]

% --- Option pass-through to article ------------------------------------------
% Accept conference / journal / technote / compsoc / compsocconf / 9pt / 10pt
% / 11pt and silently absorb them.
\DeclareOption{conference}{}
\DeclareOption{journal}{}
\DeclareOption{technote}{}
\DeclareOption{compsoc}{}
\DeclareOption{compsocconf}{}
\DeclareOption{peerreview}{}
\DeclareOption{draftcls}{}
\DeclareOption{draftclsnofoot}{}
\DeclareOption{final}{}
\DeclareOption{9pt}{}
\DeclareOption{10pt}{}
\DeclareOption{11pt}{}
\DeclareOption{12pt}{}
\DeclareOption{letterpaper}{}
\DeclareOption{a4paper}{}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% --- Base on article with the IEEE conference defaults -----------------------
\LoadClass[10pt,twocolumn,letterpaper]{article}

% IEEE-like margins (Letter paper, conference template).
\RequirePackage[letterpaper,top=0.75in,bottom=1.00in,
                left=0.625in,right=0.625in,columnsep=0.3125in]{geometry}
\RequirePackage{times}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{titlesec}

% --- Section numbering: I., II., I-A, I-B ... --------------------------------
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\thesection-\Alph{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection\arabic{subsubsection})}

% Centered, small-caps, Roman-numeral section headings.
\titleformat{\section}[block]
  {\centering\normalsize\bfseries\scshape}
  {\thesection.}{0.5em}{}
\titleformat{\subsection}[block]
  {\normalsize\itshape}
  {\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}[runin]
  {\normalsize\itshape}
  {\thesubsubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{8pt}{4pt}
\titlespacing*{\subsection}{0pt}{6pt}{2pt}
\titlespacing*{\subsubsection}{0pt}{4pt}{0.5em}

\captionsetup[figure]{font=small,labelfont=bf,name=Fig.}
\captionsetup[table]{font=small,labelfont=bf}

% --- Title block ------------------------------------------------------------
\makeatletter
\renewcommand{\@maketitle}{%
  \newpage\null
  \begin{center}%
    \let\footnote\thanks
    {\LARGE\bfseries \@title \par}\vskip 1em
    {\large \@author \par}\vskip 0.5em
    {\@date \par}%
  \end{center}\par\vspace{1ex}}

% --- IEEEtran-specific commands (compat shims) ------------------------------
\providecommand{\IEEEoverridecommandlockouts}{}
\providecommand{\IEEEPARstart}[2]{#1#2}

% Author block: \author{ \IEEEauthorblockN{Name} \IEEEauthorblockA{Affil} ... }
\newcommand{\IEEEauthorblockN}[1]{{\normalsize\bfseries #1}\par}
\newcommand{\IEEEauthorblockA}[1]{{\small\itshape #1}\par}

% IEEEkeywords environment
\newenvironment{IEEEkeywords}%
  {\vspace{0.4em}\noindent\textit{Index Terms---\space}\ignorespaces}%
  {\par\vspace{0.6em}}

\makeatother

\endinput
