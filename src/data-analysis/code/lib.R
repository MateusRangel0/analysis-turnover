read_projectdata <- function(){
  readr::read_csv(here::here("data/sip-estimates-raw.csv"), 
                  col_types = cols(
                    .default = col_character(),
                    Priority = col_double(),
                    HoursEstimate = col_double(),
                    HoursActual = col_double(),
                    DeveloperHoursActual = col_double(),
                    TaskPerformance = col_double(),
                    DeveloperPerformance = col_double()
                  ))
}