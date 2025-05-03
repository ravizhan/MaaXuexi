package main

import (
	"crypto/md5"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/hashicorp/go-version"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func CheckUpdate() (string, error) {
	resp, err := http.Get("https://update.ravi.cool/version")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP error: %s", resp.Status)
	}
	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return "", err
	}
	v, _ := data["version"].(string)
	return v, nil
}

func GenerateMetadata(dictionary string) (map[string]interface{}, error) {
	metadata := make(map[string]interface{})

	err := filepath.Walk(dictionary, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			relativePath := strings.TrimPrefix(path, dictionary)
			if relativePath == "" {
				return nil
			}
			if strings.HasPrefix(relativePath, "config") || strings.HasPrefix(relativePath, "debug") {
				return nil
			}
			parts := strings.Split(relativePath, string(filepath.Separator))
			current := metadata
			for _, part := range parts[:len(parts)-1] {
				if part == "" {
					continue
				}
				if current[part] == nil {
					current[part] = make(map[string]interface{})
				}
				current = current[part].(map[string]interface{})
			}
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()
			hash := md5.New()
			_, err = io.Copy(hash, file)
			if err != nil {
				return err
			}
			fileHash := fmt.Sprintf("%x", hash.Sum(nil))
			current[parts[len(parts)-1]] = fileHash
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return metadata, nil
}

// Powered by DeepSeek V3
func compareJSON(oldData, newData map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for key, newValue := range newData {
		oldValue, exists := oldData[key]
		if !exists {
			// 处理新增的键
			result[key] = newValue
			continue
		}
		// 递归处理嵌套map
		if newMap, ok := newValue.(map[string]interface{}); ok {
			if oldMap, ok := oldValue.(map[string]interface{}); ok {
				nestedResult := compareJSON(oldMap, newMap)
				if len(nestedResult) > 0 {
					result[key] = nestedResult
				}
			} else {
				result[key] = newValue
			}
			continue
		}
		// 处理值变更
		if fmt.Sprintf("%v", oldValue) != fmt.Sprintf("%v", newValue) {
			result[key] = newValue
		}
	}
	return result
}

func CheckOs() string {
	if goos := runtime.GOOS; goos == "windows" {
		return "windows"
	} else if goos == "linux" {
		return "ubuntu"
	} else if goos == "darwin" {
		return "macos"
	}
	return ""
}

func GetRemoteMetadata(goos string, version string) map[string]interface{} {
	resp, err := http.Get("https://update.ravi.cool/metadata?os=" + goos + "&version=" + version)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Println("Error: ", resp.Status)
		return nil
	}
	var remoteMetadata map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&remoteMetadata)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	metadata, ok := remoteMetadata["metadata"].(map[string]interface{})
	if !ok {
		fmt.Println("Error: metadata is not a map")
		return nil
	}
	return metadata
}

func Update(CurrentVersion string) {
	NewVersion, _ := CheckUpdate()
	v1, err := version.NewVersion(NewVersion)
	v2, err := version.NewVersion(CurrentVersion)
	if v1.LessThan(v2) {
		fmt.Println("Not updating, current version is newer")
	}
	localMetadata, err := GenerateMetadata("./")
	file1, _ := os.Create("local.json")
	defer file1.Close()
	encoder1 := json.NewEncoder(file1)
	encoder1.SetIndent("", "  ")
	encoder1.Encode(localMetadata)
	if err != nil {
		fmt.Println("Error:", err)
	}
	goos := CheckOs()
	remoteMetadata := GetRemoteMetadata(goos, NewVersion)
	result := compareJSON(localMetadata, remoteMetadata)
}

func main() {
	var update bool
	var v string
	flag.BoolVar(&update, "update", false, "Update directory metadata")
	flag.StringVar(&v, "v", "", "Current version")
	flag.Parse()

	if update {
		Update(v)
	}
}
