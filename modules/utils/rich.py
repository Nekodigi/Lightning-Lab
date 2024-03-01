from rich.table import Table


def makeStrTable(columns: list[str], data: list[list[str]]) -> Table:
    table = Table()
    for col in columns:
        table.add_column(col)
    for row in data:
        table.add_row(*row)
    return table
