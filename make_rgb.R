library(FITSio)
library(ggplot2)
library(dplyr)

#' This function reads 3 fits images corresponding to R, G, and B channels, normalizes their intensities, and combines them into a RGB image, saved in a .png file.
#'
#' @param r_path Path to the fits file that will be mapped to the red channel.
#' @param g_path Path to the fits file that will be mapped to the green channel.
#' @param b_path Path to the fits file that will be mapped to the blue channel.
#' @param normalize_fn A normalization function to apply to each band. If NULL, a default normalization is used. You can also pass others functions (e.g., asinh stretch)
#'
#' @return A \code{ggplot2} object representing the plot.


make_rgb_image <- function(r_path, g_path, b_path, out_path = "rgb_output.png", normalize_fn = NULL) {
  
  # Read the fits files
  g_band <- readFITS(g_path)
  r_band <- readFITS(r_path)
  i_band <- readFITS(b_path)
  
  
  R <- r_band$imDat
  G <- g_band$imDat
  B <- i_band$imDat

  # Default normalization 
  if (is.null(normalize_fn)) {
    normalize_fn <- function(x) (x - min(x)) / (max(x) - min(x))
  }
  
  Rnorm <- normalize_fn(R)
  Gnorm <- normalize_fn(G)
  Bnorm <- normalize_fn(B)
  
  # RGB dataframe
  df <- expand.grid(x = 1:ncol(R), y = 1:nrow(R)) %>%
   mutate(
    r = as.vector(Rnorm),
    g = as.vector(Gnorm),
    b = as.vector(Bnorm),
    rgb = rgb(r, g, b)
	  )
	  
  # Create ggplot object
  p <- ggplot(df, aes(x = x, y = y)) +
	 geom_raster(aes(fill = rgb)) +
	 scale_fill_identity() +
	 coord_fixed() +
	 theme_void()

  # Save the image
  ggsave(out_path, plot = p, width = 8, height = 8, dpi = 300)
  
  # Show plot
  print(p)
  
  # Return ggplot object for further use
  return(p)
  
  
  
  
# How to use:
# p <- make_rgb_image(
#  r_path = "F200W_region1.fits",
#  g_path = "cF150W_region1.fits",
#  b_path = "F115W_region1.fits",
#  out_path = "rgb_region1.png")
#
# Or, with a stretch normalization:
# stretch_asinh <- function(x, scale = 0.01) asinh(x / scale) / asinh(1 / scale)
# p <- make_rgb_image(
#  r_path = "F200W_region1.fits",
#  g_path = "cF150W_region1.fits",
#  b_path = "F115W_region1.fits",
#  out_path = "rgb_region1.png",
#  normalize_fn = stretch_asinh)
