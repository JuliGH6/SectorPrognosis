"""SectorPrognosis: Leakage-free S&P 500 sector return forecasting framework."""

SECTORS = [
    "HealthCare",
    "Materials",
    "Industrials",
    "ConsumerStaples",
    "ConsumerDiscretionary",
    "Utilities",
]

MACRO_COLS = [
    "JoblessClaimsYoY",
    "Total Jobless Claims",
    "GDPChained",
    "UnemploymentRate",
    "CPI",
    "CoreCPI",
    "CorePCE",
    "PPI",
    "ISMManufacturing",
    "ISMService",
    "ExistingHomeSales",
    "NewHomeSales",
    "EmploymentRatio",
]

HORIZON_DAYS = 250  # 1 trading year, matching original timeframe=250
