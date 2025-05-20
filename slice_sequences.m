function [in, out] = slice_sequences(data_in, data_out, seq_length, step_size)
% data_in = my_in;
% data_out = my_out;
% seq_length = 100;
% step_size = 50;

    num_samples = size(data_in,1);
    num_in_features = size(data_in,2);
    num_out_features = size(data_out,2);
    
    % Initialize output cell arrays
    in = {};
    out = {};
    
    sequence_count = 1;
    
    for start_idx = 1:step_size:(num_samples - seq_length)
        input_range = start_idx : (start_idx + seq_length - 1);
        target_idx = start_idx + seq_length;
    
        if target_idx <= num_samples
            in{sequence_count} = data_in(input_range,:);
            out{sequence_count} = data_out(max(input_range)+1,:)';

            sequence_count = sequence_count + 1;
        end
    end    
end

% figure;
% plot(input_sequences{1,1});
% hold on;
% scatter(101, output_targets{1,1});
% hold on;
% legend("In", "Out");