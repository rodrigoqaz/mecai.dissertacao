"""Utilitários de formatação numérica em padrão brasileiro (vírgula decimal) para saída LaTeX."""


def br_num(x, casas=4):
    """Formata um número com `casas` decimais e vírgula como separador decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def br_sci(x, casas=2):
    """Formata em notação científica LaTeX: mantissa com vírgula e expoente em
    \\times 10^{n}, ex.: br_sci(0.0208, 2) -> '2,08 \\times 10^{-2}'."""
    mantissa, expo = f"{x:.{casas}e}".split("e")
    return f"{mantissa.replace('.', ',')} \\times 10^{{{int(expo)}}}"
