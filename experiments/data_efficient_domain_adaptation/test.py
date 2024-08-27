def main():
    print("Running!")


if __name__ == "__main__":
    main()
def aggregate_results(results):
    out = {}
    for key, value in results.items():
      out[key] = {}
      result = np.array(list(value.values()))
      out[key]['mean'] = round(np.mean(result), 3)
      out[key]['se'] = round(np.std(result, ddof=1) / np.sqrt(len(result)), 3)
    return out
      