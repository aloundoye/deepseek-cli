use ratatui::style::Color;

fn parse_theme_color(name: &str) -> Color {
    match name.to_ascii_lowercase().as_str() {
        "black" => Color::Black,
        "red" => Color::Red,
        "green" => Color::Green,
        "yellow" => Color::Yellow,
        "blue" => Color::Blue,
        "magenta" => Color::Magenta,
        "cyan" => Color::Cyan,
        "white" => Color::White,
        "gray" | "grey" => Color::Gray,
        "darkgray" | "darkgrey" => Color::DarkGray,
        "lightred" => Color::LightRed,
        "lightgreen" => Color::LightGreen,
        "lightyellow" => Color::LightYellow,
        "lightblue" => Color::LightBlue,
        "lightmagenta" => Color::LightMagenta,
        "lightcyan" => Color::LightCyan,
        _ => Color::Cyan,
    }
}

#[derive(Debug, Clone)]
pub struct TuiTheme {
    pub primary: Color,
    pub secondary: Color,
    pub error: Color,
}

impl Default for TuiTheme {
    fn default() -> Self {
        Self {
            primary: Color::Cyan,
            secondary: Color::Yellow,
            error: Color::Red,
        }
    }
}

impl TuiTheme {
    pub fn from_config(primary: &str, secondary: &str, error: &str) -> Self {
        Self {
            primary: parse_theme_color(primary),
            secondary: parse_theme_color(secondary),
            error: parse_theme_color(error),
        }
    }
}
