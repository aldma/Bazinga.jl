function s = get_figure_style()
    % style for figure export
    s = struct();
    s.Version = '1';
    s.Format = 'eps';
    s.Preview = 'none';
    s.Width = 'auto';
    s.Height = 'auto';
    s.Units = 'centimeters';
    s.Color = 'rgb';
    s.Background = 'w';
    s.FixedFontSize = '10';
    s.ScaledFontSize = 'auto';
    s.FontMode = 'scaled';
    s.FontSizeMin = '8';
    s.FixedLineWidth = '2';
    s.ScaledLineWidth = 'auto';
    s.LineMode = 'none';
    s.LineWidthMin = '0.5';
    s.FontName = 'auto';
    s.FontWeight = 'auto';
    s.FontAngle = 'auto';
    s.FontEncoding = 'latin1';
    s.PSLevel = '3';
    s.Renderer = 'painters';
    s.Resolution = 'auto';
    s.LineStyleMap = 'none';
    s.ApplyStyle = '0';
    s.Bounds = 'tight';
    s.LockAxes = 'on';
    s.LockAxesTicks = 'off';
    s.ShowUI = 'on';
    s.SeparateText = 'off';
    return
end