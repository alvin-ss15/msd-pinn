% Simple box oscillation animation with disturbance plot
function animate_msd(in)
% Box properties
 box_width = 1;
 box_height = 1;
 y_center = 0; % vertical center of box
 gap = 2;
 
 % Create figure with tight layout
 figure('Position', [100, 100, 800, 600]);
 
 % subplot for position time history
 subplot(3,1,1);
 hold on;
 plot_h1 = plot(NaN, NaN, 'b-', 'LineWidth', 1.5); % Line for x1
 plot_h3 = plot(NaN, NaN, 'm-', 'LineWidth', 1.5); % Line for x3
 xlabel('Time (10^{-2} s)', 'Interpreter', 'tex');
 ylabel('Position (m)');
 title('Mass-Spring-Damper System');
 legend('Mass 1 (x1)', 'Mass 3 (x3)');
 time_data = zeros(1, length(in));
 x1_data = zeros(1, length(in));
 x3_data = zeros(1, length(in));
 dist_data = zeros(1, length(in));
 
 % subplot for animation
 subplot(3,1,2);
 yticklabels({});
 axis([0 10 -0.5 1.5]);
 daspect([1 1 1])
 xlabel('Position (m)');
 
 % subplot for disturbance
 subplot(3,1,3);
 hold on;
 plot_dist = plot(NaN, NaN, 'r-', 'LineWidth', 1.5); % Line for disturbance
 xlabel('Time (10^{-2} s)', 'Interpreter', 'tex');
 ylabel('Disturbance Force (N)');
 legend("Disturbance-Mass 2")
 
 % Create video writer
 v = VideoWriter('box_animation.mp4', 'MPEG-4');
 v.FrameRate = 30; % frames per second
 open(v);
 
 % Animation loop
 for i = 1:length(in)
    % Store current positions and disturbance
    time_data(1:i) = 1:i;
    x1_data(i) = in(i,1)' + gap; % x1 position
    x2_data(i) = in(i,2)' + 2*gap; % x3 position
    x3_data(i) = in(i,3)' + 3*gap; % x3 position
    dist_data(i) = in(i,7)'; % disturbance data
    
    % Update position plot
    subplot(3,1,1);
    set(plot_h1, 'XData', time_data(1:i), 'YData', x1_data(1:i));
    set(plot_h3, 'XData', time_data(1:i), 'YData', x3_data(1:i));
    % Adjust axis limits as needed
    xlim([1 length(in)]);
    min_1 = min(in(:,1:3), [], 'all')+gap;
    max_1 = max(in(:,1:3), [], 'all')+3*gap;
    ylim([min_1-0.1*abs(min_1), max_1+0.1*abs(max_1)])

    % Update animation subplot
    subplot(3,1,2);
    cla;
    % Current box position
    x1_center = in(i,1)'+gap;
    x2_center = in(i,2)'+ 2*gap;
    x3_center = in(i,3)'+ 3*gap;
    
    % Define box corners
    x1_box = [x1_center - box_width/2, x1_center + box_width/2, ...
     x1_center + box_width/2, x1_center - box_width/2];
    y_box = [y_center - box_height/2, y_center - box_height/2, ...
     y_center + box_height/2, y_center + box_height/2];
    x2_box = [x2_center - box_width/2, x2_center + box_width/2, ...
     x2_center + box_width/2, x2_center - box_width/2];
    x3_box = [x3_center - box_width/2, x3_center + box_width/2, ...
     x3_center + box_width/2, x3_center - box_width/2];
    
    % Draw the boxes
    fill(x1_box, y_box, 'blue', 'EdgeColor', 'white', 'FaceAlpha', 0.8, 'LineWidth', 2);
    hold on;
    fill(x2_box, y_box, 'red', 'EdgeColor', 'white', 'FaceAlpha', 0.8, 'LineWidth', 2);
    fill(x3_box, y_box, 'magenta', 'EdgeColor', 'white', 'FaceAlpha', 0.8, 'LineWidth', 2);
    
    % Draw lines between boxes
    % Line from wall to box 1
    line([-2, x1_center - box_width/2], ...
     [y_center, y_center], 'Color', 'black', 'LineWidth', 2);
    % Line from box 1 to box 2
    line([x1_center + box_width/2, x2_center - box_width/2], ...
     [y_center, y_center], 'Color', 'black', 'LineWidth', 2);
    % Line from box 2 to box 3
    line([x2_center + box_width/2, x3_center - box_width/2], ...
     [y_center, y_center], 'Color', 'black', 'LineWidth', 2);
    
    yticklabels({});
    axis([0 10 -0.5 1.5]);
    daspect([1 1 1])
    xlabel('Absolute Position (m)');
    
    % Update disturbance plot
    min_2 = min(in(:,7));
    max_2 = max(in(:,7));
    subplot(3,1,3);
    set(plot_dist, 'XData', time_data(1:i), 'YData', dist_data(1:i));
    xlim([1 length(in)]);
    ylim([min_2-0.25*abs(min_2), max_2+0.25*abs(max_2)]);
    
    % Capture frame
    drawnow;
    frame = getframe(gcf);
    writeVideo(v, frame);
 end
 
 close(v);
end